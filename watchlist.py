"""
watchlist.py
============
Rolling 5-day live watchlist for the Indian model.

Combines three things into one operational dashboard:

  1. The signal generator from `New_model.py::nightly_watchlist`
     (probability >= 0.65 in {bull_trend, bear_trend} after universe
     filter, scored with the production regime router).

  2. The ORB / feature-aligned break detectors from `entry_variants_v4.py`
     (now exposed as INTRADAY flags so they're actionable AT ENTRY TIME,
     not just retrospectively).

  3. Forward returns + MAE/MFE since signal, with an explicit `status`
     column that tells you exactly where each signal sits in its
     T+1..T+5 hold lifecycle.

ON LOOK-AHEAD (important)
-------------------------
Each break definition is split into TWO columns to cleanly separate
real-time entry signals from historical post-mortem flags:

  *_intraday_ts   = timestamp of the 5-min bar at which the break first
                    occurred. Filled the moment the bar closes above the
                    level. ACTIONABLE FOR ENTRY in real time.

  *_held_eod      = bool: did the close at 15:30 still sit above the
                    level? Filled only at EOD. Use this to grade whether
                    a breakout actually followed through; do NOT use it
                    as an entry trigger because EOD is unknown intraday.

For FRESH signals (signal_date == today, no T+1 yet) all forward fields
are NaN/None and `status = FRESH`. These are tomorrow's candidates.

OUTPUTS  (under BASE_DIR/watchlist/)
------------------------------------
  watchlist.parquet           current snapshot (overwrite each run)
  watchlist_history.parquet   append-only with run_ts
  watchlist.xlsx              formatted human view, conditional formatting
                              on return columns

USAGE
-----
    python watchlist.py
    python watchlist.py --window-days 5 --prob-min 0.65
    python watchlist.py --limit 100   # quick test

PROB PINNING
------------
Each row's `probability` is the value the model produced FROM THAT DAY'S
features. The script does not retroactively re-score historical rows; if
you retrain the model and re-run, only NEW rows get scored with the new
weights. Old rows already in `watchlist_history.parquet` keep their
historical probabilities.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG  (mirror New_model.py / entry_variants_v4.py)
# =============================================================================

BASE_DIR = Path(os.environ.get(
    "BASE_DIR",
    r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full",
))
PANEL_PATH = BASE_DIR / "panel_cache.parquet"
FEATURES_PATH = BASE_DIR / "features_train.json"
ROUTER_PATH = BASE_DIR / "models" / "m5_regime_router.joblib"
INTRADAY_DIR = Path(os.environ.get(
    "INTRADAY_DIR",
    r"C:\Users\karanvsi\Desktop\Pycharm\Cache\intraday_5min",
))

WATCHLIST_DIR = BASE_DIR / "watchlist"
WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_PARQUET = WATCHLIST_DIR / "watchlist.parquet"
HISTORY_PARQUET = WATCHLIST_DIR / "watchlist_history.parquet"
XLSX_PATH = WATCHLIST_DIR / "watchlist.xlsx"

IST = "Asia/Kolkata"

# Signal generation
WINDOW_DAYS = 5
PROB_MIN = 0.65
REGIMES = ["bull_trend", "bear_trend"]
MIN_CLOSE = 2.0
MIN_AVG20_VOL = 200_000

HOLDING_DAYS = 5

# ORB / breakout thresholds (mirror entry_variants_v4.py)
RANGE_EXPAND_FRAC = 0.5
DVOL_SURGE_FRAC_OF_DAY = 0.25
GAPDOWN_REJECT_PCT = -0.5

FIRST_15M_BARS = 3
FIRST_30M_BARS = 6


# =============================================================================
# 1. SIGNAL GENERATION
# =============================================================================

def generate_signals(window_days: int = WINDOW_DAYS,
                      prob_min: float = PROB_MIN) -> pd.DataFrame:
    """Score the last `window_days` of panel rows; return rows passing
    the production filter (prob >= prob_min, regime in REGIMES, universe)."""
    from joblib import load

    print(f"[signals] reading panel: {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    if panel["timestamp"].dt.tz is None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(IST)
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    panel["avg20_vol"] = (
        panel.groupby("symbol")["volume"]
             .transform(lambda s: s.rolling(20, min_periods=1).mean())
    )

    if "stock_regime" not in panel.columns:
        raise SystemExit("FATAL: panel missing stock_regime column")

    panel = panel[(panel["close"] >= MIN_CLOSE) &
                   (panel["avg20_vol"] >= MIN_AVG20_VOL)]
    panel = panel[panel["stock_regime"].isin(REGIMES)]

    all_dates = sorted(panel["timestamp"].dt.normalize().unique())
    if len(all_dates) < window_days:
        raise SystemExit(
            f"FATAL: panel has only {len(all_dates)} dates, need >= {window_days}"
        )
    cutoff = all_dates[-window_days]
    recent = panel[panel["timestamp"].dt.normalize() >= cutoff].copy()
    print(f"[signals] {len(recent):,} candidate rows in last {window_days} days "
          f"(cutoff: {cutoff.date()})")

    schema = json.loads(FEATURES_PATH.read_text())
    FEATURES = schema["features"]
    IMPUTE = {k: float(v) for k, v in schema["impute"].items()}

    print(f"[signals] loading router: {ROUTER_PATH}")
    router = load(ROUTER_PATH)
    X = recent.reindex(columns=FEATURES).copy()
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(IMPUTE.get(c, 0.0))
    recent["probability"] = router.predict_proba_by_regime(X, recent["stock_regime"])

    sigs = recent[recent["probability"] >= prob_min].copy()
    keep = ["symbol", "timestamp", "close", "high", "low", "open",
            "probability", "stock_regime"]
    sigs = sigs[keep].copy()
    sigs.columns = ["symbol", "signal_date", "prev_close", "prev_high",
                     "prev_low", "prev_open", "probability", "regime"]
    print(f"[signals] {len(sigs):,} signals at prob >= {prob_min}")
    return sigs


# =============================================================================
# 2. ROLLING-LEVEL ENRICHMENT
# =============================================================================

def enrich_rolling_levels(signals: pd.DataFrame,
                           panel_path: Path = PANEL_PATH) -> pd.DataFrame:
    """Attach prev_high_20d, prev_high_252d, prev_avg_dvol20, prev_atr14,
    prev_range_pct from panel at (symbol, signal_date)."""
    print(f"[levels] computing rolling levels from panel")
    cols_required = ["symbol", "timestamp", "high", "low", "close", "volume"]
    optional = ["D_atr14", "D_dollar_vol", "D_range_pct"]
    pn_all_cols = pd.read_parquet(panel_path).columns.tolist()
    cols_to_read = cols_required + [c for c in optional if c in pn_all_cols]
    pn = pd.read_parquet(panel_path, columns=cols_to_read)
    pn["timestamp"] = pd.to_datetime(pn["timestamp"])
    if pn["timestamp"].dt.tz is None:
        pn["timestamp"] = pn["timestamp"].dt.tz_localize(IST)
    pn = pn.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    g = pn.groupby("symbol", group_keys=False)
    pn["prev_high_20d"] = g["high"].apply(
        lambda s: s.shift(1).rolling(20, min_periods=10).max())
    pn["prev_high_252d"] = g["high"].apply(
        lambda s: s.shift(1).rolling(252, min_periods=100).max())
    if "D_dollar_vol" in pn.columns:
        pn["prev_avg_dvol20"] = g["D_dollar_vol"].apply(
            lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    else:
        dvol = pn["close"] * pn["volume"]
        pn["prev_avg_dvol20"] = (dvol.groupby(pn["symbol"])
                                       .transform(lambda s: s.shift(1)
                                                  .rolling(20, min_periods=10).mean()))
    pn["prev_atr14"] = pn.get("D_atr14", np.nan)
    if "D_range_pct" in pn.columns:
        pn["prev_range_pct"] = pn["D_range_pct"]
    else:
        pn["prev_range_pct"] = (pn["high"] - pn["low"]) / pn["close"]

    keep = ["symbol", "timestamp", "prev_high_20d", "prev_high_252d",
            "prev_avg_dvol20", "prev_atr14", "prev_range_pct"]
    pn_join = pn[keep].rename(columns={"timestamp": "signal_date"})

    return signals.merge(pn_join, on=["symbol", "signal_date"], how="left")


# =============================================================================
# 3. INTRADAY-CACHE READER + ORB DETECTORS
# =============================================================================

def _filter_market_hours(b: pd.DataFrame) -> pd.DataFrame:
    ts = b["timestamp"]
    return b[((ts.dt.hour > 9) | ((ts.dt.hour == 9) & (ts.dt.minute >= 15))) &
            ((ts.dt.hour < 15) | ((ts.dt.hour == 15) & (ts.dt.minute <= 30)))]


def _read_intraday_for_symbol(symbol: str) -> Optional[pd.DataFrame]:
    fp = INTRADAY_DIR / f"{symbol}.parquet"
    if not fp.exists():
        return None
    try:
        bars = pd.read_parquet(fp)
    except Exception:
        return None
    if bars.empty:
        return None
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    if bars["timestamp"].dt.tz is None:
        bars["timestamp"] = bars["timestamp"].dt.tz_localize(IST)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    return _filter_market_hours(bars)


@dataclass
class OrbDetection:
    t1_orh_15min: float = np.nan
    t1_orl_15min: float = np.nan

    # Intraday flags (entry-time actionable). Stored as the timestamp of
    # the bar at which the break first occurred, NaT/None if no break.
    orb_break_15m_intraday_ts: Optional[pd.Timestamp] = None
    dist20h_break_intraday_ts: Optional[pd.Timestamp] = None
    dist52wh_break_intraday_ts: Optional[pd.Timestamp] = None

    # Held-EOD historical annotations (NOT entry-actionable).
    # None until T+1 completes; True/False once the day is closed.
    orb_break_15m_held_eod: Optional[bool] = None
    dist20h_break_held_eod: Optional[bool] = None
    dist52wh_break_held_eod: Optional[bool] = None

    range_expand_15m: bool = False
    dvol_surge_30m: bool = False

    confluence_count_intraday: int = 0
    confluence_count_held_eod: int = 0

    t1_first_30m_dvol: float = np.nan
    gap_pct: float = np.nan


def detect_orb_for_signal(sig: pd.Series,
                           bars_by_day: Dict[pd.Timestamp, pd.DataFrame],
                           today: pd.Timestamp) -> OrbDetection:
    """Run all 4 atomic detectors on T+1 of a single signal."""
    out = OrbDetection()
    sd = pd.Timestamp(sig["signal_date"]).normalize()

    after_days = sorted([d for d in bars_by_day.keys() if d > sd])
    if not after_days:
        return out
    t1 = after_days[0]
    if t1 > today:
        return out
    t1_bars = bars_by_day[t1]
    if t1_bars.empty:
        return out

    if len(t1_bars) >= FIRST_15M_BARS:
        first15 = t1_bars.head(FIRST_15M_BARS)
        out.t1_orh_15min = float(first15["high"].max())
        out.t1_orl_15min = float(first15["low"].min())

    open_t1 = float(t1_bars.iloc[0]["open"])
    prev_close = float(sig.get("prev_close", np.nan))
    if not pd.isna(prev_close) and prev_close > 0:
        out.gap_pct = (open_t1 / prev_close - 1.0) * 100.0
    if not pd.isna(out.gap_pct) and out.gap_pct < GAPDOWN_REJECT_PCT:
        return out

    closes = t1_bars["close"].to_numpy(dtype=float)
    timestamps = t1_bars["timestamp"].to_list()
    last_close = float(t1_bars.iloc[-1]["close"])

    # T+1 day complete iff its last bar is at-or-after 15:25 (15:25-15:30 bar)
    last_ts = pd.Timestamp(t1_bars.iloc[-1]["timestamp"])
    t1_complete = last_ts.time() >= dt.time(15, 25)

    prev_high_20d = sig.get("prev_high_20d", np.nan)
    prev_high_252d = sig.get("prev_high_252d", np.nan)
    prev_range_pct = sig.get("prev_range_pct", np.nan)
    prev_avg_dvol20 = sig.get("prev_avg_dvol20", np.nan)

    # 1. 15-min ORB break (intraday: first close above ORH after first 3 bars)
    if not pd.isna(out.t1_orh_15min):
        for i in range(FIRST_15M_BARS - 1, len(closes)):
            if closes[i] > out.t1_orh_15min:
                out.orb_break_15m_intraday_ts = timestamps[i]
                break
        if t1_complete:
            out.orb_break_15m_held_eod = bool(last_close > out.t1_orh_15min)

    # 2. DIST20H break
    if not pd.isna(prev_high_20d):
        for i in range(len(closes)):
            if closes[i] > prev_high_20d:
                out.dist20h_break_intraday_ts = timestamps[i]
                break
        if t1_complete:
            out.dist20h_break_held_eod = bool(last_close > prev_high_20d)

    # 3. DIST52WH break
    if not pd.isna(prev_high_252d):
        for i in range(len(closes)):
            if closes[i] > prev_high_252d:
                out.dist52wh_break_intraday_ts = timestamps[i]
                break
        if t1_complete:
            out.dist52wh_break_held_eod = bool(last_close > prev_high_252d)

    # 4. RANGE_EXPAND
    if (not pd.isna(prev_range_pct) and not pd.isna(prev_close)
            and len(t1_bars) >= FIRST_15M_BARS):
        first15 = t1_bars.head(FIRST_15M_BARS)
        rng = float(first15["high"].max() - first15["low"].min())
        threshold = RANGE_EXPAND_FRAC * float(prev_range_pct) * float(prev_close)
        if rng >= threshold:
            out.range_expand_15m = True

    # 5. DVOL_SURGE
    if (not pd.isna(prev_avg_dvol20) and len(t1_bars) >= FIRST_30M_BARS
            and "volume" in t1_bars.columns):
        first30 = t1_bars.head(FIRST_30M_BARS)
        dvol_cum = float((first30["close"] * first30["volume"]).sum())
        out.t1_first_30m_dvol = dvol_cum
        if dvol_cum >= DVOL_SURGE_FRAC_OF_DAY * float(prev_avg_dvol20):
            out.dvol_surge_30m = True

    # Confluence counts
    out.confluence_count_intraday = int(sum([
        out.orb_break_15m_intraday_ts is not None,
        out.dist20h_break_intraday_ts is not None,
        out.range_expand_15m,
        out.dvol_surge_30m,
    ]))
    if t1_complete:
        out.confluence_count_held_eod = int(sum([
            out.orb_break_15m_held_eod is True,
            out.dist20h_break_held_eod is True,
            out.range_expand_15m,
            out.dvol_surge_30m,
        ]))
    return out


def add_orb_flags(signals: pd.DataFrame,
                   today: pd.Timestamp) -> pd.DataFrame:
    """For each signal, attach all 4 atomic detectors as columns.
    Reads each symbol's intraday parquet ONCE for performance."""
    print(f"[orb] processing {len(signals):,} signals across "
          f"{signals['symbol'].nunique():,} symbols")

    rows = signals.copy().reset_index(drop=True)
    new_cols = [
        "t1_orh_15min", "t1_orl_15min",
        "orb_break_15m_intraday_ts", "orb_break_15m_held_eod",
        "dist20h_break_intraday_ts", "dist20h_break_held_eod",
        "dist52wh_break_intraday_ts", "dist52wh_break_held_eod",
        "range_expand_15m", "dvol_surge_30m",
        "confluence_count_intraday", "confluence_count_held_eod",
        "t1_first_30m_dvol", "gap_pct",
    ]
    for c in new_cols:
        rows[c] = pd.NA

    n_symbols = rows["symbol"].nunique()
    n_no_cache = 0
    for sym_idx, (sym, sym_sigs) in enumerate(rows.groupby("symbol"), 1):
        if sym_idx % 50 == 0:
            print(f"   [orb] {sym_idx}/{n_symbols} symbols processed")
        bars = _read_intraday_for_symbol(sym)
        if bars is None or bars.empty:
            n_no_cache += 1
            continue
        bars_by_day = {d: g.reset_index(drop=True)
                        for d, g in bars.groupby(bars["timestamp"].dt.normalize())}
        for idx, sig in sym_sigs.iterrows():
            res = detect_orb_for_signal(sig, bars_by_day, today)
            for k in new_cols:
                rows.at[idx, k] = getattr(res, k)
    print(f"[orb] symbols with no intraday cache: {n_no_cache:,}")
    return rows


# =============================================================================
# 4. RETURNS + MAE/MFE + STATUS
# =============================================================================

def add_returns_and_excursions(signals: pd.DataFrame,
                                 panel_path: Path = PANEL_PATH,
                                 today: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if today is None:
        today = pd.Timestamp.now(tz=IST).normalize()

    print(f"[returns] computing forward returns from panel")
    pn = pd.read_parquet(
        panel_path,
        columns=["symbol", "timestamp", "open", "high", "low", "close"],
    )
    pn["timestamp"] = pd.to_datetime(pn["timestamp"])
    if pn["timestamp"].dt.tz is None:
        pn["timestamp"] = pn["timestamp"].dt.tz_localize(IST)
    pn["date_norm"] = pn["timestamp"].dt.normalize()

    rows = signals.copy().reset_index(drop=True)
    new_cols = ["t1_open", "current_close", "current_close_date",
                 "ret_from_prev_close_pct", "ret_from_t1_open_pct",
                 "mae_so_far_pct", "mfe_so_far_pct",
                 "days_held", "status"]
    for c in new_cols:
        rows[c] = pd.NA

    for sym, sym_sigs in rows.groupby("symbol"):
        sym_panel = pn[pn["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        if sym_panel.empty:
            for idx in sym_sigs.index:
                rows.at[idx, "status"] = "FRESH"
                rows.at[idx, "days_held"] = 0
            continue

        for idx, sig in sym_sigs.iterrows():
            sd = pd.Timestamp(sig["signal_date"]).normalize()
            forward = sym_panel[sym_panel["date_norm"] > sd]
            if forward.empty:
                rows.at[idx, "days_held"] = 0
                rows.at[idx, "status"] = "FRESH"
                continue

            t1_row = forward.iloc[0]
            t1_open_v = float(t1_row["open"])
            rows.at[idx, "t1_open"] = t1_open_v

            held = forward.head(HOLDING_DAYS).copy()
            last_row = held.iloc[-1]
            cc = float(last_row["close"])
            rows.at[idx, "current_close"] = cc
            rows.at[idx, "current_close_date"] = last_row["date_norm"]

            prev_close = float(sig["prev_close"])
            if prev_close > 0:
                rows.at[idx, "ret_from_prev_close_pct"] = (cc / prev_close - 1) * 100
            if t1_open_v > 0:
                rows.at[idx, "ret_from_t1_open_pct"] = (cc / t1_open_v - 1) * 100

            if prev_close > 0:
                low_min = float(held["low"].min())
                high_max = float(held["high"].max())
                rows.at[idx, "mae_so_far_pct"] = (low_min / prev_close - 1) * 100
                rows.at[idx, "mfe_so_far_pct"] = (high_max / prev_close - 1) * 100

            n_days = len(held)
            rows.at[idx, "days_held"] = n_days
            rows.at[idx, "status"] = (
                "EXPIRED" if len(forward) >= HOLDING_DAYS else f"T+{n_days}"
            )
    return rows


# =============================================================================
# 5. WRITE OUTPUTS
# =============================================================================

OUTPUT_COLUMN_ORDER = [
    # Core identity
    "signal_date", "symbol", "regime", "probability", "status", "days_held",
    # Entry references
    "prev_close", "prev_high", "prev_low", "prev_open",
    "t1_open", "t1_orh_15min", "t1_orl_15min",
    # Rolling levels
    "prev_high_20d", "prev_high_252d", "prev_avg_dvol20", "prev_atr14",
    "prev_range_pct",
    # Breakout flags (intraday = entry-time, held_eod = retrospective)
    "gap_pct",
    "orb_break_15m_intraday_ts",   "orb_break_15m_held_eod",
    "dist20h_break_intraday_ts",   "dist20h_break_held_eod",
    "dist52wh_break_intraday_ts",  "dist52wh_break_held_eod",
    "range_expand_15m", "dvol_surge_30m",
    "t1_first_30m_dvol",
    "confluence_count_intraday", "confluence_count_held_eod",
    # Performance
    "ret_from_prev_close_pct", "ret_from_t1_open_pct",
    "mae_so_far_pct", "mfe_so_far_pct",
    "current_close", "current_close_date",
]


def write_outputs(watchlist: pd.DataFrame, run_ts: pd.Timestamp):
    cols = [c for c in OUTPUT_COLUMN_ORDER if c in watchlist.columns]
    extra = [c for c in watchlist.columns if c not in cols]
    df = watchlist[cols + extra].copy()

    status_priority = {"FRESH": 0, "T+1": 1, "T+2": 2, "T+3": 3,
                        "T+4": 4, "T+5": 5, "EXPIRED": 6}
    df["__sp"] = df["status"].map(status_priority).fillna(99)
    df = df.sort_values(["__sp", "signal_date", "probability"],
                          ascending=[True, False, False]).drop(columns="__sp")

    df.to_parquet(SNAPSHOT_PARQUET, index=False)
    print(f"[out] {SNAPSHOT_PARQUET}")

    df_hist = df.copy()
    df_hist.insert(0, "run_ts", run_ts)
    if HISTORY_PARQUET.exists():
        try:
            old = pd.read_parquet(HISTORY_PARQUET)
            combined = pd.concat([old, df_hist], ignore_index=True)
        except Exception as e:
            print(f"[warn] history file unreadable ({e}); starting fresh")
            combined = df_hist
    else:
        combined = df_hist
    combined.to_parquet(HISTORY_PARQUET, index=False)
    print(f"[out] {HISTORY_PARQUET}  (cumulative rows: {len(combined):,})")

    try:
        import openpyxl
        from openpyxl.styles import Font, Alignment
        from openpyxl.formatting.rule import ColorScaleRule
        with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="watchlist", index=False)
            ws = xw.book["watchlist"]
            for col in ws.columns:
                max_len = max((len(str(c.value)) for c in col), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 30)
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center")
            for ret_col in ("ret_from_prev_close_pct", "ret_from_t1_open_pct"):
                if ret_col in df.columns:
                    ci = df.columns.get_loc(ret_col) + 1
                    letter = openpyxl.utils.get_column_letter(ci)
                    cell_range = f"{letter}2:{letter}{len(df) + 1}"
                    rule = ColorScaleRule(
                        start_type="num", start_value=-10, start_color="F8696B",
                        mid_type="num", mid_value=0, mid_color="FFFFFF",
                        end_type="num", end_value=10, end_color="63BE7B",
                    )
                    ws.conditional_formatting.add(cell_range, rule)
        print(f"[out] {XLSX_PATH}")
    except Exception as e:
        print(f"[warn] xlsx skipped: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=WINDOW_DAYS,
                    help=f"how many trading days back to scan (default {WINDOW_DAYS})")
    ap.add_argument("--prob-min", type=float, default=PROB_MIN,
                    help=f"minimum model probability (default {PROB_MIN})")
    ap.add_argument("--limit", type=int, default=0,
                    help="if > 0, only process first N signals (testing)")
    args = ap.parse_args()

    run_ts = pd.Timestamp.now(tz=IST)
    today = run_ts.normalize()
    print(f"[main] run_ts: {run_ts.isoformat()}")
    print(f"[main] today (IST): {today.date()}")

    sigs = generate_signals(window_days=args.window_days, prob_min=args.prob_min)
    if args.limit > 0:
        sigs = sigs.head(args.limit).reset_index(drop=True)
        print(f"[main] LIMITED to first {len(sigs):,} signals")

    sigs = enrich_rolling_levels(sigs)
    sigs = add_orb_flags(sigs, today=today)
    sigs = add_returns_and_excursions(sigs, today=today)

    write_outputs(sigs, run_ts=run_ts)

    print("\n===== WATCHLIST SUMMARY =====")
    if "status" in sigs.columns:
        print("Status counts:")
        print(sigs["status"].value_counts().to_string())
    print(f"\nTotal signals      : {len(sigs):,}")
    n_fresh = int((sigs["status"] == "FRESH").sum()) if "status" in sigs.columns else 0
    print(f"FRESH (act tomorrow): {n_fresh}")

    if n_fresh > 0:
        fresh = sigs[sigs["status"] == "FRESH"][
            ["symbol", "regime", "probability", "prev_close",
             "prev_high_20d", "prev_high_252d"]
        ].sort_values("probability", ascending=False)
        print(f"\nFRESH names (top by probability):")
        with pd.option_context("display.max_rows", 30, "display.float_format",
                                lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)):
            print(fresh.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
