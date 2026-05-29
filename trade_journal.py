"""
trade_journal.py
================
Automatic trade journal for the Indian model.

Where `watchlist.py` answers "what should I watch and how is it doing?",
the trade journal answers "for a given entry policy, what trades did I
actually take, when did they exit, and what did they make?".

It builds on `watchlist.py` (imports its signal generation, panel-level
rolling enrichment, and intraday cache reader) and adds a per-trade
lifecycle with entry detection, OCO exit simulation, realized P&L,
MAE/MFE since entry, time-elapsed, and R-multiple.

ONE ROW PER (signal x variant) TRADE
------------------------------------
For each signal in the rolling window and each configured entry variant,
the journal:
  1. detects the intraday entry (the moment the variant's condition fires)
  2. walks forward from entry through a TP / SL / T+5-close OCO ladder
  3. records entry, exit, realized return, excursions, and clock-time

ENTRY VARIANTS (mirror entry_variants_v4.py, intraday-actionable)
-----------------------------------------------------------------
  NAIVE_T1_OPEN        enter at T+1 open
  ORB_15_BREAK         enter at first 5m close > T+1 first-15m high
  DIST20H_BREAK        enter at first 5m close > prev 20-day high
  DIST52WH_BREAK       enter at first 5m close > prev 252-day high

EXIT LADDER (configurable; default TP=+3%, SL=-3%, time-stop T+5 close)
----------------------------------------------------------------------
  exit_reason in {TP_HIT, SL_HIT, T5_CLOSE, ACTIVE}
  - stop checked before target on the same bar (conservative)
  - ACTIVE for trades still open (T+5 not yet reached / entry just fired)

ON LOOK-AHEAD
-------------
Entry detection uses only bars up to and including the entry bar. Exit
detection walks forward from the entry bar; for ACTIVE trades the exit
fields are NaN/None until the bars exist. No exit field is ever filled
from a bar that hasn't printed.

OUTPUTS  (under BASE_DIR/trade_journal/)
----------------------------------------
  trade_journal.parquet            current snapshot (overwrite each run)
  trade_journal_history.parquet    append-only with run_ts
  trade_journal.csv                human-friendly mirror
  trade_journal.xlsx               formatted, conditional formatting on P&L
  trade_journal_summary.csv        per-variant aggregates (win%, avg R, etc.)

USAGE
-----
    python trade_journal.py
    python trade_journal.py --window-days 10 --prob-min 0.70
    python trade_journal.py --tp 5 --sl -3 --variants NAIVE_T1_OPEN,DIST20H_BREAK
    python trade_journal.py --limit 200    # quick test
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Reuse everything shared with the watchlist so conventions never drift.
import watchlist as wl

IST = wl.IST
HOLDING_DAYS = wl.HOLDING_DAYS
FIRST_15M_BARS = wl.FIRST_15M_BARS
GAPDOWN_REJECT_PCT = wl.GAPDOWN_REJECT_PCT

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = wl.BASE_DIR
JOURNAL_DIR = BASE_DIR / "trade_journal"
JOURNAL_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_PARQUET = JOURNAL_DIR / "trade_journal.parquet"
HISTORY_PARQUET = JOURNAL_DIR / "trade_journal_history.parquet"
CSV_PATH = JOURNAL_DIR / "trade_journal.csv"
XLSX_PATH = JOURNAL_DIR / "trade_journal.xlsx"
SUMMARY_CSV = JOURNAL_DIR / "trade_journal_summary.csv"

# Costs (mirror entry_variants_v4.py: 25 bps round-trip + 5 bps each side)
COST_BPS_ROUND_TRIP = 25
SLIPPAGE_BPS_PER_SIDE = 5
TOTAL_COST_PCT = (COST_BPS_ROUND_TRIP + 2 * SLIPPAGE_BPS_PER_SIDE) / 100.0  # ~0.35%

# Default exit ladder
DEFAULT_TP_PCT = 3.0
DEFAULT_SL_PCT = -3.0

# Entry variants this journal can log
DEFAULT_VARIANTS = ["NAIVE_T1_OPEN", "ORB_15_BREAK", "DIST20H_BREAK"]


# =============================================================================
# ENTRY DETECTION  (intraday-actionable, no look-ahead)
# =============================================================================

@dataclass
class Entry:
    fired: bool = False
    bar_idx: int = -1            # index into the T+1..T+5 bar arrays
    ts: Optional[pd.Timestamp] = None
    price: float = np.nan
    method: str = ""
    day_offset: int = -1         # 1 = T+1, 2 = T+2, ...
    gap_pct: float = np.nan


def _day1_indices(bar_days: np.ndarray) -> np.ndarray:
    return np.where(bar_days == 0)[0]


def detect_entry(variant: str,
                 bar_ts: List[pd.Timestamp],
                 bar_opens: np.ndarray,
                 bar_highs: np.ndarray,
                 bar_lows: np.ndarray,
                 bar_closes: np.ndarray,
                 bar_days: np.ndarray,
                 sig: pd.Series) -> Entry:
    """Find the entry bar for one variant. Uses only data at/under the
    entry bar -- safe for live use."""
    out = Entry(method=variant)
    d1 = _day1_indices(bar_days)
    if len(d1) == 0:
        return out

    open_t1 = float(bar_opens[d1[0]])
    prev_close = float(sig.get("prev_close", np.nan))
    if not pd.isna(prev_close) and prev_close > 0:
        out.gap_pct = (open_t1 / prev_close - 1.0) * 100.0

    # All non-naive variants honour the gap-down safety gate.
    gate_ok = pd.isna(out.gap_pct) or out.gap_pct >= GAPDOWN_REJECT_PCT

    if variant == "NAIVE_T1_OPEN":
        i = int(d1[0])
        out.fired = True
        out.bar_idx = i
        out.ts = bar_ts[i]
        out.price = open_t1
        out.day_offset = 1
        return out

    if not gate_ok:
        return out

    if variant == "ORB_15_BREAK":
        orh = sig.get("t1_orh_15min")
        if orh is None or pd.isna(orh):
            # derive from first 15m of T+1 if not precomputed
            if len(d1) >= FIRST_15M_BARS:
                orh = float(np.max(bar_highs[d1[:FIRST_15M_BARS]]))
            else:
                return out
        for i in range(FIRST_15M_BARS - 1, len(bar_closes)):
            if int(bar_days[i]) < 0 or int(bar_days[i]) >= HOLDING_DAYS:
                continue
            if bar_closes[i] > orh:
                out.fired = True
                out.bar_idx = i
                out.ts = bar_ts[i]
                out.price = float(bar_closes[i])
                out.day_offset = int(bar_days[i]) + 1
                return out
        return out

    if variant == "DIST20H_BREAK":
        lvl = sig.get("prev_high_20d")
        if lvl is None or pd.isna(lvl):
            return out
        for i in range(len(bar_closes)):
            if int(bar_days[i]) < 0 or int(bar_days[i]) >= HOLDING_DAYS:
                continue
            if bar_closes[i] > lvl:
                out.fired = True
                out.bar_idx = i
                out.ts = bar_ts[i]
                out.price = float(bar_closes[i])
                out.day_offset = int(bar_days[i]) + 1
                return out
        return out

    if variant == "DIST52WH_BREAK":
        lvl = sig.get("prev_high_252d")
        if lvl is None or pd.isna(lvl):
            return out
        for i in range(len(bar_closes)):
            if int(bar_days[i]) < 0 or int(bar_days[i]) >= HOLDING_DAYS:
                continue
            if bar_closes[i] > lvl:
                out.fired = True
                out.bar_idx = i
                out.ts = bar_ts[i]
                out.price = float(bar_closes[i])
                out.day_offset = int(bar_days[i]) + 1
                return out
        return out

    return out


# =============================================================================
# EXIT SIMULATION  (TP / SL / T+5-close OCO)
# =============================================================================

@dataclass
class Exit:
    reason: str = "ACTIVE"           # TP_HIT / SL_HIT / T5_CLOSE / ACTIVE
    bar_idx: int = -1
    ts: Optional[pd.Timestamp] = None
    price: float = np.nan
    hold_bars: int = 0
    mae_since_entry_pct: float = np.nan
    mfe_since_entry_pct: float = np.nan


def simulate_exit(entry: Entry,
                  bar_ts: List[pd.Timestamp],
                  bar_highs: np.ndarray,
                  bar_lows: np.ndarray,
                  bar_closes: np.ndarray,
                  bar_days: np.ndarray,
                  tp_pct: float,
                  sl_pct: float,
                  holding_days: int = HOLDING_DAYS) -> Exit:
    """
    Walk forward from the entry bar through the hold window and resolve
    the OCO exit. Stop is checked before target on the same bar.

    Returns an Exit with reason ACTIVE if the hold window hasn't fully
    printed yet AND neither TP nor SL has been touched -- i.e. the trade
    is genuinely still open.
    """
    out = Exit()
    e_idx = entry.bar_idx
    e_px = entry.price
    if e_idx < 0 or pd.isna(e_px) or e_px <= 0:
        return out

    entry_day = int(bar_days[e_idx])
    end_day = entry_day + holding_days       # exclusive upper day bound
    tp_px = e_px * (1 + tp_pct / 100.0)
    sl_px = e_px * (1 + sl_pct / 100.0)

    n = len(bar_closes)
    raw_mae = 0.0
    raw_mfe = 0.0
    last_idx_in_window = e_idx
    window_complete = False

    for j in range(e_idx + 1, n):
        if int(bar_days[j]) > end_day - 1:
            window_complete = True
            break
        last_idx_in_window = j
        lo_pct = (bar_lows[j] / e_px - 1) * 100
        hi_pct = (bar_highs[j] / e_px - 1) * 100
        raw_mae = min(raw_mae, lo_pct)
        raw_mfe = max(raw_mfe, hi_pct)

        # stop first (conservative)
        if bar_lows[j] <= sl_px:
            out.reason = "SL_HIT"
            out.bar_idx = j
            out.ts = bar_ts[j]
            out.price = sl_px
            out.hold_bars = j - e_idx
            out.mae_since_entry_pct = raw_mae
            out.mfe_since_entry_pct = raw_mfe
            return out
        if bar_highs[j] >= tp_px:
            out.reason = "TP_HIT"
            out.bar_idx = j
            out.ts = bar_ts[j]
            out.price = tp_px
            out.hold_bars = j - e_idx
            out.mae_since_entry_pct = raw_mae
            out.mfe_since_entry_pct = raw_mfe
            return out

    # Neither TP nor SL fired within the bars we have.
    if window_complete or (last_idx_in_window > e_idx and
                            int(bar_days[last_idx_in_window]) >= end_day - 1):
        # Hold window fully printed -> time-stop exit at last in-window close
        out.reason = "T5_CLOSE"
        out.bar_idx = last_idx_in_window
        out.ts = bar_ts[last_idx_in_window]
        out.price = float(bar_closes[last_idx_in_window])
        out.hold_bars = last_idx_in_window - e_idx
        out.mae_since_entry_pct = raw_mae
        out.mfe_since_entry_pct = raw_mfe
        return out

    # Trade still open (window not yet complete).
    out.reason = "ACTIVE"
    out.bar_idx = last_idx_in_window
    out.ts = bar_ts[last_idx_in_window] if last_idx_in_window > e_idx else entry.ts
    out.price = (float(bar_closes[last_idx_in_window])
                 if last_idx_in_window > e_idx else e_px)
    out.hold_bars = last_idx_in_window - e_idx
    out.mae_since_entry_pct = raw_mae
    out.mfe_since_entry_pct = raw_mfe
    return out


# =============================================================================
# PER-SIGNAL -> TRADE ROWS
# =============================================================================

def journal_one_signal(sig: pd.Series,
                        bars_by_day: Dict[pd.Timestamp, pd.DataFrame],
                        today: pd.Timestamp,
                        variants: List[str],
                        tp_pct: float,
                        sl_pct: float) -> List[Dict]:
    """Produce one journal row per variant that fires for this signal."""
    sd = pd.Timestamp(sig["signal_date"]).normalize()
    after_days = sorted([d for d in bars_by_day.keys() if d > sd])
    if not after_days:
        return []

    # Assemble contiguous T+1..T+HOLDING bar arrays (same layout as ORB code).
    hold_days = after_days[: HOLDING_DAYS]
    frames = [bars_by_day[d] for d in hold_days if not bars_by_day[d].empty]
    if not frames:
        return []
    bars = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if bars.empty:
        return []

    day_index = {d.normalize(): i for i, d in enumerate(hold_days)}
    bar_ts = list(bars["timestamp"])
    bar_opens = bars["open"].to_numpy(dtype=float)
    bar_highs = bars["high"].to_numpy(dtype=float)
    bar_lows = bars["low"].to_numpy(dtype=float)
    bar_closes = bars["close"].to_numpy(dtype=float)
    bar_days = np.array([day_index.get(pd.Timestamp(t).normalize(), -1)
                         for t in bar_ts], dtype=int)

    rows: List[Dict] = []
    for variant in variants:
        entry = detect_entry(variant, bar_ts, bar_opens, bar_highs,
                             bar_lows, bar_closes, bar_days, sig)
        if not entry.fired:
            rows.append({
                "signal_date": sd, "symbol": sig["symbol"],
                "regime": sig.get("regime"), "probability": sig.get("probability"),
                "variant": variant, "entry_fired": False,
                "status": "NO_ENTRY",
                "gap_pct": entry.gap_pct,
            })
            continue

        ex = simulate_exit(entry, bar_ts, bar_highs, bar_lows, bar_closes,
                           bar_days, tp_pct, sl_pct)

        # realized return after round-trip cost (only when closed)
        if ex.reason == "ACTIVE":
            realized_ret = np.nan
            realized_net = np.nan
        else:
            gross = (ex.price / entry.price - 1.0) * 100.0
            realized_ret = gross
            realized_net = gross - TOTAL_COST_PCT

        # time elapsed (clock hours from entry bar to exit bar)
        if entry.ts is not None and ex.ts is not None:
            elapsed_h = (pd.Timestamp(ex.ts) - pd.Timestamp(entry.ts)).total_seconds() / 3600.0
        else:
            elapsed_h = np.nan

        # R-multiple: realized return divided by the risk taken (|SL distance|)
        risk_pct = abs(sl_pct)
        r_multiple = (realized_net / risk_pct) if (risk_pct > 0 and not pd.isna(realized_net)) else np.nan

        rows.append({
            "signal_date": sd,
            "symbol": sig["symbol"],
            "regime": sig.get("regime"),
            "probability": sig.get("probability"),
            "variant": variant,
            "entry_fired": True,
            "status": ex.reason,                       # TP_HIT/SL_HIT/T5_CLOSE/ACTIVE
            "gap_pct": entry.gap_pct,
            # entry
            "entry_ts": entry.ts,
            "entry_price": entry.price,
            "entry_method": entry.method,
            "entry_day_offset": entry.day_offset,       # 1=T+1, ...
            "prev_close": float(sig.get("prev_close", np.nan)),
            "prev_high_20d": sig.get("prev_high_20d"),
            "prev_high_252d": sig.get("prev_high_252d"),
            "t1_orh_15min": sig.get("t1_orh_15min"),
            # exit
            "exit_ts": ex.ts,
            "exit_price": ex.price,
            "exit_reason": ex.reason,
            "hold_bars": ex.hold_bars,
            "hold_days": round(ex.hold_bars * 5.0 / 375.0, 2) if ex.hold_bars else 0.0,  # 375 min/session
            "time_elapsed_hours": round(elapsed_h, 2) if not pd.isna(elapsed_h) else np.nan,
            # performance
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "realized_ret_pct": realized_ret,
            "realized_net_pct": realized_net,
            "r_multiple": r_multiple,
            "mae_since_entry_pct": ex.mae_since_entry_pct,
            "mfe_since_entry_pct": ex.mfe_since_entry_pct,
            "slippage_from_prev_close_pct":
                (entry.price / float(sig["prev_close"]) - 1) * 100
                if sig.get("prev_close") else np.nan,
        })
    return rows


# =============================================================================
# BUILD JOURNAL
# =============================================================================

def build_journal(signals: pd.DataFrame,
                  today: pd.Timestamp,
                  variants: List[str],
                  tp_pct: float,
                  sl_pct: float) -> pd.DataFrame:
    """Read each symbol's intraday cache once, journal all its signals."""
    print(f"[journal] {len(signals):,} signals  x  {len(variants)} variants")
    print(f"[journal] exit ladder: TP=+{tp_pct}%  SL={sl_pct}%  time-stop=T+{HOLDING_DAYS}")

    rows: List[Dict] = []
    n_symbols = signals["symbol"].nunique()
    n_no_cache = 0
    for i, (sym, sym_sigs) in enumerate(signals.groupby("symbol"), 1):
        if i % 50 == 0:
            print(f"   [journal] {i}/{n_symbols} symbols, {len(rows):,} rows")
        bars = wl._read_intraday_for_symbol(sym)
        if bars is None or bars.empty:
            n_no_cache += 1
            continue
        bars_by_day = {d: g.reset_index(drop=True)
                       for d, g in bars.groupby(bars["timestamp"].dt.normalize())}
        for _, sig in sym_sigs.iterrows():
            rows.extend(journal_one_signal(sig, bars_by_day, today,
                                           variants, tp_pct, sl_pct))
    print(f"[journal] symbols with no intraday cache: {n_no_cache:,}")
    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY
# =============================================================================

def summarize(journal: pd.DataFrame) -> pd.DataFrame:
    """Per (variant x regime) realized stats over CLOSED trades only."""
    closed = journal[journal["status"].isin(["TP_HIT", "SL_HIT", "T5_CLOSE"])].copy()
    if closed.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in closed.groupby(["variant", "regime"], dropna=False):
        variant, regime = keys
        n = len(g)
        net = g["realized_net_pct"]
        rows.append({
            "variant": variant,
            "regime": regime,
            "n_closed": n,
            "n_active": int(((journal["variant"] == variant) &
                             (journal["regime"] == regime) &
                             (journal["status"] == "ACTIVE")).sum()),
            "win_pct": round(100.0 * (net > 0).mean(), 2),
            "tp_hit_pct": round(100.0 * (g["status"] == "TP_HIT").mean(), 2),
            "sl_hit_pct": round(100.0 * (g["status"] == "SL_HIT").mean(), 2),
            "t5_close_pct": round(100.0 * (g["status"] == "T5_CLOSE").mean(), 2),
            "mean_net_pct": round(net.mean(), 3),
            "median_net_pct": round(net.median(), 3),
            "mean_r_multiple": round(g["r_multiple"].mean(), 3),
            "mean_hold_days": round(g["hold_days"].mean(), 2),
            "mean_time_hours": round(g["time_elapsed_hours"].mean(), 2),
            "mean_mae_pct": round(g["mae_since_entry_pct"].mean(), 3),
            "mean_mfe_pct": round(g["mfe_since_entry_pct"].mean(), 3),
            "total_net_pct": round(net.sum(), 2),
        })
    return pd.DataFrame(rows).sort_values(["variant", "regime"]).reset_index(drop=True)


# =============================================================================
# OUTPUT (reuse watchlist's Excel sanitiser to avoid the tz blank-file bug)
# =============================================================================

COLUMN_ORDER = [
    "signal_date", "symbol", "regime", "probability", "variant",
    "status", "entry_fired",
    "entry_ts", "entry_price", "entry_method", "entry_day_offset",
    "exit_ts", "exit_price", "exit_reason",
    "hold_bars", "hold_days", "time_elapsed_hours",
    "tp_pct", "sl_pct",
    "realized_ret_pct", "realized_net_pct", "r_multiple",
    "mae_since_entry_pct", "mfe_since_entry_pct",
    "gap_pct", "slippage_from_prev_close_pct",
    "prev_close", "prev_high_20d", "prev_high_252d", "t1_orh_15min",
]


def write_outputs(journal: pd.DataFrame, summary: pd.DataFrame,
                  run_ts: pd.Timestamp):
    cols = [c for c in COLUMN_ORDER if c in journal.columns]
    extra = [c for c in journal.columns if c not in cols]
    df = journal[cols + extra].copy()

    status_priority = {"ACTIVE": 0, "TP_HIT": 1, "T5_CLOSE": 2,
                       "SL_HIT": 3, "NO_ENTRY": 4}
    df["__sp"] = df["status"].map(status_priority).fillna(9)
    df = df.sort_values(["__sp", "signal_date", "variant", "probability"],
                        ascending=[True, False, True, False]).drop(columns="__sp")

    df.to_parquet(SNAPSHOT_PARQUET, index=False)
    print(f"[out] {SNAPSHOT_PARQUET}")
    df.to_csv(CSV_PATH, index=False)
    print(f"[out] {CSV_PATH}")

    # append-only history
    df_hist = df.copy()
    df_hist.insert(0, "run_ts", run_ts)
    if HISTORY_PARQUET.exists():
        try:
            old = pd.read_parquet(HISTORY_PARQUET)
            combined = pd.concat([old, df_hist], ignore_index=True)
        except Exception as e:
            print(f"[warn] history unreadable ({e}); starting fresh")
            combined = df_hist
    else:
        combined = df_hist
    combined.to_parquet(HISTORY_PARQUET, index=False)
    print(f"[out] {HISTORY_PARQUET}  (cumulative rows: {len(combined):,})")

    if not summary.empty:
        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"[out] {SUMMARY_CSV}")

    # XLSX (sanitise tz-aware datetimes -> openpyxl-safe)
    try:
        import openpyxl
        from openpyxl.styles import Font, Alignment
        from openpyxl.formatting.rule import ColorScaleRule
        df_xl = wl._sanitize_for_excel(df)
        with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as xw:
            df_xl.to_excel(xw, sheet_name="trades", index=False)
            if not summary.empty:
                wl._sanitize_for_excel(summary).to_excel(
                    xw, sheet_name="summary", index=False)
            ws = xw.sheets["trades"]
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 30)
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center")
            for ret_col in ("realized_net_pct", "r_multiple"):
                if ret_col in df_xl.columns:
                    ci = df_xl.columns.get_loc(ret_col) + 1
                    letter = openpyxl.utils.get_column_letter(ci)
                    rng = f"{letter}2:{letter}{len(df_xl) + 1}"
                    ws.conditional_formatting.add(rng, ColorScaleRule(
                        start_type="num", start_value=-5, start_color="F8696B",
                        mid_type="num", mid_value=0, mid_color="FFFFFF",
                        end_type="num", end_value=5, end_color="63BE7B"))
        check = pd.read_excel(XLSX_PATH, sheet_name="trades")
        if len(check) != len(df_xl):
            raise RuntimeError(f"xlsx round-trip mismatch: {len(df_xl)} vs {len(check)}")
        print(f"[out] {XLSX_PATH}  (verified: {len(check)} rows)")
    except Exception as e:
        import traceback
        print(f"[ERROR] xlsx write FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"        Parquet + CSV are complete at {SNAPSHOT_PARQUET} / {CSV_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Automatic trade journal.")
    ap.add_argument("--window-days", type=int, default=wl.WINDOW_DAYS)
    ap.add_argument("--prob-min", type=float, default=wl.PROB_MIN)
    ap.add_argument("--tp", type=float, default=DEFAULT_TP_PCT,
                    help="take-profit percent (default +3)")
    ap.add_argument("--sl", type=float, default=DEFAULT_SL_PCT,
                    help="stop-loss percent, negative (default -3)")
    ap.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS),
                    help="comma-separated entry variants")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run_ts = pd.Timestamp.now(tz=IST)
    today = run_ts.normalize()
    print(f"[main] run_ts {run_ts.isoformat()}  variants={variants}")

    # Reuse watchlist's signal generation + rolling-level enrichment.
    sigs = wl.generate_signals(window_days=args.window_days, prob_min=args.prob_min)
    if args.limit > 0:
        sigs = sigs.head(args.limit).reset_index(drop=True)
        print(f"[main] LIMITED to first {len(sigs):,} signals")
    sigs = wl.enrich_rolling_levels(sigs)

    journal = build_journal(sigs, today, variants, args.tp, args.sl)
    if journal.empty:
        print("[main] no journal rows produced (no intraday data?).")
        return

    summary = summarize(journal)
    write_outputs(journal, summary, run_ts)

    # console teaser
    print("\n===== TRADE JOURNAL SUMMARY =====")
    closed = journal[journal["status"].isin(["TP_HIT", "SL_HIT", "T5_CLOSE"])]
    active = journal[journal["status"] == "ACTIVE"]
    no_entry = journal[journal["status"] == "NO_ENTRY"]
    print(f"  rows total : {len(journal):,}")
    print(f"  closed     : {len(closed):,}")
    print(f"  active     : {len(active):,}")
    print(f"  no-entry   : {len(no_entry):,}")
    if not summary.empty:
        show = ["variant", "regime", "n_closed", "win_pct", "tp_hit_pct",
                "sl_hit_pct", "mean_net_pct", "mean_r_multiple",
                "mean_hold_days", "mean_time_hours"]
        print("\n" + summary[show].to_string(index=False))


if __name__ == "__main__":
    main()
