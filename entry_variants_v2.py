"""
entry_variants_v2.py
====================
Cheap-entry alternatives to ORB confirmation.

THESIS
------
ORB pays ~2-3% in entry slippage to filter losers via "close > prev_high",
but the model's actual target is `top20_vs_bot20_5d` (ATR-normalized 5-day
rank). The model selects stocks that *don't break down*, not stocks that
*break out hard*. A handful of cheaper, direction-only filters should
capture most of ORB's selection benefit at a fraction of the entry cost.

This module is a self-contained extension of ``ORB execution.py`` that:
  * Re-uses the ``extracted_paths_v2.parquet`` artifact that script
    produces (so no intraday re-fetch is needed).
  * Adds 7 new entry variants (see ENTRY_VARIANTS below) plus 5 ORB
    variants from the original.
  * Synthesizes 3 hybrid policies that pick a different variant per
    probability bucket, using the rule we validated empirically.
  * Writes a comparison CSV / xlsx with per-trade and portfolio-level
    metrics for every variant and every hybrid.

VARIANTS ADDED
--------------
  GREEN_15m_t1            -> green first 15-min bar; enter at 09:30 close
  ABOVE_PREVCLOSE_OPEN    -> enter at T+1 open if open >= prev_close
  NO_GAPDOWN_OPEN         -> enter at T+1 open if gap_pct >= -0.5%
  VWAP_RECLAIM_30m        -> first 5-min close >= intraday VWAP by 09:45
  GREEN_FIRSTHOUR         -> 09:15-10:15 cum return > 0; enter at 10:15
  PREVCLOSE_HOLD_EOD      -> T+1 close >= prev_close; enter at T+1 close
  LIMIT_AT_PREVCLOSE      -> buy limit at prev_close (gap-down filter)

USAGE
-----
    python entry_variants_v2.py --reuse-v2

    # Or override input/output paths via env vars:
    EVV2_PATHS=/path/to/extracted_paths_v2.parquet \
    EVV2_OUT=/path/to/out_dir \
        python entry_variants_v2.py

INPUTS
------
The parquet must have the same schema as `_extract_paths` in
``ORB execution.py``: per-signal rows with bar_timestamps / bar_opens /
bar_highs / bar_lows / bar_closes / bar_days arrays + day-level OHLC and
per-day OR levels.

OUTPUTS  (under <OUT_DIR>/)
--------
  per_trade_extended.parquet       per-(signal, variant) outcome
  variants_comparison.csv          headline metrics for every variant
  hybrid_policies.csv              the 3 hybrid policies head-to-head
  variants_summary.xlsx            all of the above as tabs
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG  (mirror ORB execution.py)
# =============================================================================

BASE_DIR = Path(
    os.environ.get("EVV2_BASE_DIR", r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
)
V2_PATHS_FILE = Path(
    os.environ.get("EVV2_PATHS", str(BASE_DIR / "orb_machine_results_v2" / "extracted_paths_v2.parquet"))
)
OUT_DIR = Path(os.environ.get("EVV2_OUT", str(BASE_DIR / "entry_variants_v2")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

IST = "Asia/Kolkata"

# ---- signal filter / buckets ------------------------------------------------
SIGNAL_PROB_MIN = 0.65
SIGNAL_REGIMES = ["bull_trend", "bear_trend"]
PROB_BUCKETS = [(0.65, 0.70), (0.70, 0.75), (0.75, 0.85), (0.85, 1.01)]

# ---- holding / extraction window -------------------------------------------
HOLDING_DAYS = 5
EXTENDED_DAYS = 10

# ---- costs ------------------------------------------------------------------
COST_BPS_ROUND_TRIP = 25
SLIPPAGE_BPS_PER_SIDE = 5
TOTAL_COST_PCT = (COST_BPS_ROUND_TRIP + 2 * SLIPPAGE_BPS_PER_SIDE) / 100.0  # ~0.35%

# ---- TP / SL ladder ---------------------------------------------------------
TP_LEVELS_PCT = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
SL_LEVELS_PCT = [-1.0, -2.0, -3.0, -5.0]

# ---- range definitions for ORB variants ------------------------------------
RANGE_DEFINITIONS = {
    "5min":  {"end_time": "09:20", "bars_needed": 1},
    "15min": {"end_time": "09:30", "bars_needed": 3},
    "30min": {"end_time": "09:45", "bars_needed": 6},
    "60min": {"end_time": "10:15", "bars_needed": 12},
}

# ---- intraday parameters for new variants ----------------------------------
GAPDOWN_THRESH_PCT = -0.5     # name disqualified at the open
VWAP_DEADLINE_BARS = 6        # 6 x 5-min = 30 min, i.e. by 09:45
FIRST_15M_BARS = 3            # 09:15-09:30
FIRST_HOUR_BARS = 12          # 09:15-10:15

# =============================================================================
# VARIANT CATALOG
# =============================================================================

VARIANTS: Dict[str, Dict] = {
    # --- ORB baselines (for comparison) ---------------------------------------
    "ORB_15_t1_only": {
        "label": "15-min ORB, close above, T+1 only",
        "engine": "orb",
        "range_def": "15min", "watch_mode": "t1_only", "confirm": "close_above",
    },
    "ORB_15_anyday_5d": {
        "label": "15-min ORB, close above, T+1..T+5 (static T+1 range)",
        "engine": "orb",
        "range_def": "15min", "watch_mode": "static_t1_5d", "confirm": "close_above",
    },
    "ORB_15_held_anyday": {
        "label": "15-min ORB, held EOD, T+1..T+5 (static)",
        "engine": "orb",
        "range_def": "15min", "watch_mode": "static_t1_5d", "confirm": "close_held_eod",
    },

    # --- Naive baseline (no filter, T+1 open entry) --------------------------
    "NAIVE_T1_OPEN": {
        "label": "Naive T+1 open entry, no filter",
        "engine": "alt",
        "confirm": "naive_open",
    },

    # --- Cheap direction filters (the 7 new variants) ------------------------
    "GREEN_15m_t1": {
        "label": "Green 09:15-09:30 candle, enter at 09:30 close",
        "engine": "alt",
        "confirm": "first_bar_green",
    },
    "ABOVE_PREVCLOSE_OPEN": {
        "label": "Enter at T+1 open if open >= prev_close",
        "engine": "alt",
        "confirm": "open_above_prev",
    },
    "NO_GAPDOWN_OPEN": {
        "label": f"Enter at T+1 open if gap_pct >= {GAPDOWN_THRESH_PCT}%",
        "engine": "alt",
        "confirm": "no_gapdown",
    },
    "VWAP_RECLAIM_30m": {
        "label": "Enter on first 5m close >= intraday VWAP by 09:45",
        "engine": "alt",
        "confirm": "vwap_reclaim",
    },
    "GREEN_FIRSTHOUR": {
        "label": "09:15-10:15 cum return > 0, enter at 10:15 close",
        "engine": "alt",
        "confirm": "first_hour_green",
    },
    "PREVCLOSE_HOLD_EOD": {
        "label": "Enter at T+1 close if T+1 close >= prev_close",
        "engine": "alt",
        "confirm": "prevclose_hold_eod",
    },
    "LIMIT_AT_PREVCLOSE": {
        "label": "Buy limit at prev_close on T+1, reject if gap_down > 0.5%",
        "engine": "alt",
        "confirm": "limit_at_prevclose",
    },

    # --- ORB variants from original (kept for completeness) ------------------
    "ORB_15_anyday_dyn": {
        "label": "15-min ORB, close above, T+1..T+5 (dynamic each day)",
        "engine": "orb",
        "range_def": "15min", "watch_mode": "dynamic_each_day", "confirm": "close_above",
    },
    "ORB_5_t1_only": {
        "label": "5-min ORB, close above, T+1 only",
        "engine": "orb",
        "range_def": "5min", "watch_mode": "t1_only", "confirm": "close_above",
    },
}

# =============================================================================
# OUTCOME DATACLASS
# =============================================================================

@dataclass
class TradeOutcome:
    triggered: bool = False
    entry_idx: int = -1
    entry_time: Optional[pd.Timestamp] = None
    entry_price: float = np.nan
    entry_day_offset: int = -1
    ref_high: float = np.nan
    ref_low: float = np.nan
    mae_pct: float = np.nan
    mfe_pct: float = np.nan
    fwd_return_to_t5_close_pct: float = np.nan
    oco: Optional[Dict[Tuple[float, str], Dict]] = None


# =============================================================================
# ORB ENTRY (kept compatible with ORB execution.py)
# =============================================================================

def _find_breakout(bar_highs, bar_lows, bar_closes, bar_days,
                   variant, sig, range_bars_needed) -> Tuple[int, float, float, float]:
    """Return (entry_idx, entry_price, ref_high, ref_low) or (-1, ...)."""
    watch_mode = variant["watch_mode"]
    confirm = variant["confirm"]
    range_def = variant["range_def"]

    if watch_mode == "t1_only":
        watch_max = 1
        dynamic = False
    elif watch_mode == "static_t1_5d":
        watch_max = HOLDING_DAYS
        dynamic = False
    elif watch_mode == "dynamic_each_day":
        watch_max = HOLDING_DAYS
        dynamic = True
    else:
        return -1, np.nan, np.nan, np.nan

    if not dynamic:
        orh = sig.get(f"t1_orh_{range_def}")
        orl = sig.get(f"t1_orl_{range_def}")
        if pd.isna(orh) or pd.isna(orl) or orh <= orl:
            return -1, np.nan, np.nan, np.nan

    n = len(bar_closes)
    for i in range(n):
        d = int(bar_days[i])
        if d < 0 or d >= watch_max:
            continue
        if dynamic:
            day_label = d + 1
            day_orh = sig.get(f"d{day_label}_orh_{range_def}")
            day_orl = sig.get(f"d{day_label}_orl_{range_def}")
            if pd.isna(day_orh) or pd.isna(day_orl) or day_orh <= day_orl:
                continue
            same_day_before = int(np.sum(bar_days[:i] == d))
            if same_day_before < range_bars_needed:
                continue
            cur_h, cur_l = day_orh, day_orl
        else:
            if d == 0:
                same_day_before = int(np.sum(bar_days[:i] == 0))
                if same_day_before < range_bars_needed:
                    continue
            cur_h, cur_l = orh, orl

        triggered = False
        if confirm == "close_above":
            triggered = bar_closes[i] > cur_h
        elif confirm == "touch_above":
            triggered = bar_highs[i] > cur_h
        elif confirm == "close_held_eod":
            if bar_closes[i] > cur_h:
                same_day = np.where(bar_days == d)[0]
                if len(same_day) and bar_closes[same_day[-1]] > cur_h:
                    triggered = True

        if triggered:
            return i, float(bar_closes[i]), float(cur_h), float(cur_l)
    return -1, np.nan, np.nan, np.nan


# =============================================================================
# ALTERNATIVE ENTRIES (the new stuff)
# =============================================================================

def _day1_idx(bar_days: np.ndarray) -> np.ndarray:
    """Indices of all bars on T+1 (entry day, day_index = 0)."""
    return np.where(bar_days == 0)[0]


def _typical_price(highs, lows, closes) -> np.ndarray:
    return (highs + lows + closes) / 3.0


def _intraday_vwap(highs, lows, closes, k_inclusive: int) -> float:
    """
    Bar-count-weighted VWAP proxy from day-1 bars [0..k_inclusive].
    No volume column is in the bar arrays so we use equal weights on
    typical price (the same convention the original `compute_vpoc` falls
    back to). This is a defensible approximation for filtering purposes.
    """
    tp = _typical_price(highs[: k_inclusive + 1], lows[: k_inclusive + 1], closes[: k_inclusive + 1])
    return float(np.mean(tp))


def _ref_levels_from_entry(entry_price: float) -> Tuple[float, float]:
    """
    Provide (ref_high, ref_low) for the OCO ladder when a non-ORB variant
    triggers. We use +/- 1% bands as a neutral substitute for ORB high/low,
    which only affects the `range_low` and `two_range` SL columns. The TP
    and SL_PCT_LEVELS columns are unchanged.
    """
    return entry_price * 1.01, entry_price * 0.99


def _find_entry_alt(bar_opens, bar_highs, bar_lows, bar_closes, bar_days,
                    variant, sig) -> Tuple[int, float, float, float]:
    """
    Non-ORB entries. Returns (entry_idx, entry_price, ref_high, ref_low).
    Always operates only on T+1 bars (bar_days == 0). Returns (-1, ...) if
    the variant's condition is not met.
    """
    confirm = variant["confirm"]
    d1 = _day1_idx(bar_days)
    if len(d1) == 0:
        return -1, np.nan, np.nan, np.nan

    open_t1 = float(bar_opens[d1[0]])
    prev_close = float(sig.get("prev_close", np.nan))

    # ----- naive (no filter) -------------------------------------------------
    if confirm == "naive_open":
        i = int(d1[0])
        return i, open_t1, *_ref_levels_from_entry(open_t1)

    # ----- open-side filters (zero entry slippage) ---------------------------
    if confirm == "open_above_prev":
        if pd.isna(prev_close) or open_t1 < prev_close:
            return -1, np.nan, np.nan, np.nan
        i = int(d1[0])
        return i, open_t1, *_ref_levels_from_entry(open_t1)

    if confirm == "no_gapdown":
        if pd.isna(prev_close):
            return -1, np.nan, np.nan, np.nan
        gap_pct = (open_t1 / prev_close - 1.0) * 100.0
        if gap_pct < GAPDOWN_THRESH_PCT:
            return -1, np.nan, np.nan, np.nan
        i = int(d1[0])
        return i, open_t1, *_ref_levels_from_entry(open_t1)

    # ----- first-15-min direction filter -------------------------------------
    if confirm == "first_bar_green":
        if len(d1) < FIRST_15M_BARS:
            return -1, np.nan, np.nan, np.nan
        bars = d1[:FIRST_15M_BARS]
        bar_open = float(bar_opens[bars[0]])
        bar_close = float(bar_closes[bars[-1]])
        if bar_close <= bar_open:
            return -1, np.nan, np.nan, np.nan
        i = int(bars[-1])
        return i, bar_close, *_ref_levels_from_entry(bar_close)

    # ----- VWAP reclaim ------------------------------------------------------
    if confirm == "vwap_reclaim":
        deadline = min(VWAP_DEADLINE_BARS, len(d1))
        if deadline < 1:
            return -1, np.nan, np.nan, np.nan
        d1_h = bar_highs[d1[:deadline]]
        d1_l = bar_lows[d1[:deadline]]
        d1_c = bar_closes[d1[:deadline]]
        for k in range(deadline):
            vwap_k = _intraday_vwap(d1_h, d1_l, d1_c, k)
            if d1_c[k] >= vwap_k and k >= 1:
                # require at least 1 prior bar so vwap is not just the open
                i = int(d1[k])
                px = float(d1_c[k])
                return i, px, *_ref_levels_from_entry(px)
        return -1, np.nan, np.nan, np.nan

    # ----- first-hour direction filter ---------------------------------------
    if confirm == "first_hour_green":
        if len(d1) < FIRST_HOUR_BARS:
            return -1, np.nan, np.nan, np.nan
        bars = d1[:FIRST_HOUR_BARS]
        bar_open = float(bar_opens[bars[0]])
        bar_close = float(bar_closes[bars[-1]])
        if bar_close <= bar_open:
            return -1, np.nan, np.nan, np.nan
        i = int(bars[-1])
        return i, bar_close, *_ref_levels_from_entry(bar_close)

    # ----- end-of-day-1 hold-above-prev-close --------------------------------
    if confirm == "prevclose_hold_eod":
        if pd.isna(prev_close):
            return -1, np.nan, np.nan, np.nan
        last = int(d1[-1])
        c = float(bar_closes[last])
        if c < prev_close:
            return -1, np.nan, np.nan, np.nan
        return last, c, *_ref_levels_from_entry(c)

    # ----- buy limit at prev_close (with gap-down rejection) -----------------
    if confirm == "limit_at_prevclose":
        if pd.isna(prev_close):
            return -1, np.nan, np.nan, np.nan
        gap_pct = (open_t1 / prev_close - 1.0) * 100.0
        if gap_pct < GAPDOWN_THRESH_PCT:
            return -1, np.nan, np.nan, np.nan
        # If price already at/below limit at open, fill at open
        if open_t1 <= prev_close:
            i = int(d1[0])
            return i, open_t1, *_ref_levels_from_entry(open_t1)
        # Otherwise walk T+1 bars looking for low <= prev_close
        for i in d1:
            if float(bar_lows[i]) <= prev_close:
                return int(i), prev_close, *_ref_levels_from_entry(prev_close)
        return -1, np.nan, np.nan, np.nan

    return -1, np.nan, np.nan, np.nan


# =============================================================================
# FORWARD WALK (OCO ladder, MAE/MFE) -- copied from ORB execution.py
# =============================================================================

def _walk_oco(bar_highs, bar_lows, bar_closes, bar_days,
              entry_idx: int, entry_price: float, orl: float,
              orh: float, holding_days: int) -> Tuple[Dict, float, float, float]:
    """
    Walk forward `holding_days` from entry_idx. Returns:
      oco_dict, raw_mae_pct, raw_mfe_pct, fwd_to_t5_close_pct

    The OCO dict has keys (tp_pct, sl_label) where sl_label is either a
    string (e.g. "-3.0", "range_low", "two_range").
    """
    n = len(bar_highs)
    end_day = int(bar_days[entry_idx]) + holding_days
    raw_mae = 0.0
    raw_mfe = 0.0
    last_idx = entry_idx
    for j in range(entry_idx + 1, n):
        if bar_days[j] > end_day - 1:
            break
        last_idx = j
        lo_pct = (bar_lows[j] / entry_price - 1) * 100
        hi_pct = (bar_highs[j] / entry_price - 1) * 100
        raw_mae = min(raw_mae, lo_pct)
        raw_mfe = max(raw_mfe, hi_pct)
    fwd_to_t5_close = (bar_closes[last_idx] / entry_price - 1) * 100

    oco: Dict[Tuple[float, str], Dict] = {}
    sl_special = [("range_low", orl),
                  ("two_range", entry_price - 2 * (orh - orl))]
    sl_pct_levels = [(f"{lvl:.1f}", entry_price * (1 + lvl / 100.0)) for lvl in SL_LEVELS_PCT]
    sl_levels_all = sl_pct_levels + [(name, px) for name, px in sl_special if px < entry_price]

    for tp_pct in TP_LEVELS_PCT:
        tp_px = entry_price * (1 + tp_pct / 100)
        for sl_name, sl_px in sl_levels_all:
            hit_tp = False
            hit_sl = False
            exit_idx = last_idx
            for j in range(entry_idx + 1, last_idx + 1):
                if bar_lows[j] <= sl_px:
                    hit_sl = True
                    exit_idx = j
                    break
                if bar_highs[j] >= tp_px:
                    hit_tp = True
                    exit_idx = j
                    break
            if hit_tp:
                ret = tp_pct
            elif hit_sl:
                ret = (sl_px / entry_price - 1) * 100
            else:
                ret = fwd_to_t5_close
            oco[(tp_pct, sl_name)] = {
                "hit_tp": hit_tp, "hit_sl": hit_sl,
                "ret_pct": ret,
                "bars_to_exit": exit_idx - entry_idx,
            }
    return oco, raw_mae, raw_mfe, fwd_to_t5_close


# =============================================================================
# SIMULATE ONE (signal x variant)
# =============================================================================

def simulate_signal(sig: pd.Series, variant: Dict) -> TradeOutcome:
    out = TradeOutcome()
    bts = sig.get("bar_timestamps")
    if not isinstance(bts, (list, np.ndarray)) or len(bts) < 20:
        return out

    bar_opens = np.asarray(sig["bar_opens"], dtype=float)
    bar_highs = np.asarray(sig["bar_highs"], dtype=float)
    bar_lows = np.asarray(sig["bar_lows"], dtype=float)
    bar_closes = np.asarray(sig["bar_closes"], dtype=float)
    bar_days = np.asarray(sig["bar_days"], dtype=int)

    engine = variant.get("engine", "orb")

    if engine == "orb":
        rd = variant["range_def"]
        rbn = RANGE_DEFINITIONS[rd]["bars_needed"]
        e_idx, e_px, ref_hi, ref_lo = _find_breakout(
            bar_highs, bar_lows, bar_closes, bar_days, variant, sig, rbn,
        )
    else:
        e_idx, e_px, ref_hi, ref_lo = _find_entry_alt(
            bar_opens, bar_highs, bar_lows, bar_closes, bar_days, variant, sig,
        )

    if e_idx < 0:
        return out

    out.triggered = True
    out.entry_idx = e_idx
    out.entry_price = e_px
    out.entry_day_offset = int(bar_days[e_idx]) + 1
    out.ref_high, out.ref_low = ref_hi, ref_lo

    oco, mae, mfe, fwd = _walk_oco(
        bar_highs, bar_lows, bar_closes, bar_days,
        e_idx, e_px, ref_lo, ref_hi, HOLDING_DAYS,
    )
    out.oco = oco
    out.mae_pct = mae
    out.mfe_pct = mfe
    out.fwd_return_to_t5_close_pct = fwd
    return out


# =============================================================================
# RUN ALL VARIANTS  -> per_trade DataFrame
# =============================================================================

def run_all(paths_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    n_total = len(paths_df)
    print(f"[sim] {n_total:,} signals  x  {len(VARIANTS)} variants  =  "
          f"{n_total * len(VARIANTS):,} simulations")

    for var_id, variant in VARIANTS.items():
        triggered = 0
        for i, sig in enumerate(paths_df.itertuples(index=False)):
            if i and i % 5000 == 0:
                print(f"   {var_id}: {i}/{n_total}  triggered={triggered}")
            sd = sig._asdict()
            res = simulate_signal(pd.Series(sd), variant)
            base = {
                "variant": var_id,
                "engine": variant.get("engine", "orb"),
                "symbol": sd["symbol"],
                "signal_date": sd["signal_date"],
                "regime": sd["regime"],
                "probability": sd["probability"],
                "prob_bucket": sd.get("prob_bucket", ""),
                "prev_close": sd["prev_close"],
                "triggered": res.triggered,
                "entry_day_offset": res.entry_day_offset,
                "entry_price": res.entry_price,
                "ref_high": res.ref_high, "ref_low": res.ref_low,
                "slippage_from_signal_pct":
                    (res.entry_price / sd["prev_close"] - 1) * 100
                    if res.triggered else np.nan,
                "mae_pct": res.mae_pct,
                "mfe_pct": res.mfe_pct,
                "fwd_return_to_t5_close_pct": res.fwd_return_to_t5_close_pct,
            }
            if res.oco:
                triggered += 1
                for (tp, sl), o in res.oco.items():
                    base[f"tp{tp:g}_sl{sl}_ret"] = o["ret_pct"]
                    base[f"tp{tp:g}_sl{sl}_hittp"] = int(o["hit_tp"])
                    base[f"tp{tp:g}_sl{sl}_hitsl"] = int(o["hit_sl"])
            rows.append(base)
        rate = 100.0 * triggered / max(n_total, 1)
        print(f"   {var_id}: trigger rate {triggered}/{n_total} = {rate:.1f}%")
    return pd.DataFrame(rows)


# =============================================================================
# AGGREGATIONS
# =============================================================================

def _portfolio_metrics(trades: pd.DataFrame, ret_col: str = "fwd_return_to_t5_close_pct") -> Dict:
    """
    De-overlapped daily-basket portfolio: average all trade returns sharing
    the same signal_date into a single basket return; then annualize.
    """
    if trades.empty or trades[ret_col].isna().all():
        return {"port_avg_basket_ret_pct": np.nan,
                "port_ann_sharpe": np.nan,
                "port_max_dd_pct": np.nan,
                "port_days": 0}
    trades = trades.copy()
    trades[ret_col] = trades[ret_col] - TOTAL_COST_PCT
    daily = trades.groupby("signal_date")[ret_col].mean().sort_index()
    cum = (1 + daily / 100).cumprod()
    rolling_max = cum.cummax()
    dd = (cum / rolling_max - 1) * 100
    mu = daily.mean()
    sd = daily.std(ddof=1)
    sharpe = (mu / sd * np.sqrt(252.0 / max(HOLDING_DAYS, 1))) if sd and sd > 0 else np.nan
    return {
        "port_avg_basket_ret_pct": float(mu),
        "port_ann_sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "port_max_dd_pct": float(dd.min()),
        "port_days": int(len(daily)),
    }


def variant_summary(per_trade: pd.DataFrame) -> pd.DataFrame:
    """
    Per (variant x regime x prob_bucket): trigger rate, per-trade metrics,
    portfolio metrics. The compact comparison table.
    """
    rows: List[Dict] = []
    grp_cols = ["variant", "regime", "prob_bucket"]
    for keys, sub in per_trade.groupby(grp_cols):
        var_id, regime, bucket = keys
        n_sig = len(sub)
        n_trig = int(sub["triggered"].sum())
        trig_rate = 100.0 * n_trig / max(n_sig, 1)
        taken = sub[sub["triggered"]].copy()

        if not taken.empty:
            mean_ret = float(taken["fwd_return_to_t5_close_pct"].mean())
            med_ret = float(taken["fwd_return_to_t5_close_pct"].median())
            win_rate = 100.0 * float((taken["fwd_return_to_t5_close_pct"] > 0).mean())
            mean_mae = float(taken["mae_pct"].mean())
            mean_mfe = float(taken["mfe_pct"].mean())
            mean_slip = float(taken["slippage_from_signal_pct"].mean())
            net_ret = mean_ret - TOTAL_COST_PCT
        else:
            mean_ret = med_ret = win_rate = mean_mae = mean_mfe = mean_slip = net_ret = np.nan

        port = _portfolio_metrics(taken)

        rows.append({
            "variant": var_id, "regime": regime, "prob_bucket": bucket,
            "n_signals": n_sig, "n_taken": n_trig, "trigger_pct": trig_rate,
            "mean_ret_pct": mean_ret,
            "median_ret_pct": med_ret,
            "win_pct": win_rate,
            "mean_net_pct": net_ret,
            "mean_mae_pct": mean_mae,
            "mean_mfe_pct": mean_mfe,
            "mean_slippage_from_signal_pct": mean_slip,
            **port,
        })
    return pd.DataFrame(rows)


def lift_vs_naive(per_trade: pd.DataFrame) -> pd.DataFrame:
    """
    For every variant, compute (variant_portfolio_ret - naive_portfolio_ret)
    on the same set of signal-dates, per regime x bucket.
    """
    naive = per_trade[per_trade["variant"] == "NAIVE_T1_OPEN"].copy()
    if naive.empty:
        return pd.DataFrame()
    naive_taken = naive[naive["triggered"]].copy()
    naive_taken["net"] = naive_taken["fwd_return_to_t5_close_pct"] - TOTAL_COST_PCT
    naive_basket = (
        naive_taken.groupby(["regime", "prob_bucket", "signal_date"])["net"]
        .mean().rename("naive_basket").reset_index()
    )

    rows: List[Dict] = []
    for var_id, sub in per_trade[per_trade["variant"] != "NAIVE_T1_OPEN"].groupby("variant"):
        for keys, ssub in sub.groupby(["regime", "prob_bucket"]):
            regime, bucket = keys
            taken = ssub[ssub["triggered"]].copy()
            if taken.empty:
                continue
            taken["net"] = taken["fwd_return_to_t5_close_pct"] - TOTAL_COST_PCT
            v_basket = (
                taken.groupby("signal_date")["net"].mean().rename("v_basket").reset_index()
            )
            n_basket = naive_basket[(naive_basket["regime"] == regime) &
                                    (naive_basket["prob_bucket"] == bucket)]
            merged = v_basket.merge(n_basket, on="signal_date", how="inner")
            if merged.empty:
                continue
            lift = (merged["v_basket"] - merged["naive_basket"]).mean()
            rows.append({
                "variant": var_id,
                "regime": regime,
                "prob_bucket": bucket,
                "n_overlap_days": len(merged),
                "v_basket_mean_net_pct": float(merged["v_basket"].mean()),
                "naive_basket_mean_net_pct": float(merged["naive_basket"].mean()),
                "lift_vs_naive_pct": float(lift),
            })
    return pd.DataFrame(rows)


# =============================================================================
# HYBRID POLICIES
# =============================================================================

HYBRID_POLICIES = {
    # Bucket -> variant to use for that bucket
    "HYBRID_GREEN15_THEN_NAIVE": {
        "[0.65,0.70)": "GREEN_15m_t1",
        "[0.70,0.75)": "GREEN_15m_t1",
        "[0.75,0.85)": "NAIVE_T1_OPEN",
        "[0.85,1.01)": "NAIVE_T1_OPEN",
    },
    "HYBRID_VWAP_THEN_NAIVE": {
        "[0.65,0.70)": "VWAP_RECLAIM_30m",
        "[0.70,0.75)": "VWAP_RECLAIM_30m",
        "[0.75,0.85)": "NAIVE_T1_OPEN",
        "[0.85,1.01)": "NAIVE_T1_OPEN",
    },
    # The previous-Claude reference policy: ORB_held below 0.75, naive above
    "HYBRID_ORBHELD_THEN_NAIVE": {
        "[0.65,0.70)": "ORB_15_held_anyday",
        "[0.70,0.75)": "ORB_15_held_anyday",
        "[0.75,0.85)": "NAIVE_T1_OPEN",
        "[0.85,1.01)": "NAIVE_T1_OPEN",
    },
}


def synthesize_hybrid_policies(per_trade: pd.DataFrame) -> pd.DataFrame:
    """
    For each hybrid policy, build a synthetic per_trade table where each
    signal is taken from the variant the policy picks for its bucket. Then
    compute variant_summary on the union.
    """
    pieces: List[pd.DataFrame] = []
    for hyb_id, mapping in HYBRID_POLICIES.items():
        for bucket, var_id in mapping.items():
            sub = per_trade[(per_trade["variant"] == var_id) &
                            (per_trade["prob_bucket"] == bucket)].copy()
            if sub.empty:
                continue
            sub["variant"] = hyb_id
            sub["component_variant"] = var_id
            pieces.append(sub)
    if not pieces:
        return pd.DataFrame()
    hybrid_pt = pd.concat(pieces, ignore_index=True)
    return hybrid_pt


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-v2", action="store_true",
                    help="reuse extracted_paths_v2.parquet from the v2 ORB run")
    ap.add_argument("--paths-file", type=str, default=str(V2_PATHS_FILE))
    ap.add_argument("--prob-min", type=float, default=SIGNAL_PROB_MIN)
    args = ap.parse_args()

    paths_path = Path(args.paths_file)
    if not paths_path.exists():
        raise SystemExit(
            f"FATAL: paths parquet not found: {paths_path}\n"
            "Run `ORB execution.py` first (it produces this file), or set\n"
            "EVV2_PATHS or --paths-file to the existing extract."
        )
    print(f"[main] reading paths: {paths_path}")
    paths = pd.read_parquet(paths_path)
    paths = paths[paths["probability"] >= args.prob_min].reset_index(drop=True)
    print(f"[main] {len(paths):,} signals at prob >= {args.prob_min}")

    per_trade = run_all(paths)
    pt_path = OUT_DIR / "per_trade_extended.parquet"
    per_trade.to_parquet(pt_path, index=False)
    print(f"[out] {pt_path}")

    # standard variants summary
    summary = variant_summary(per_trade)
    summary_path = OUT_DIR / "variants_comparison.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[out] {summary_path}")

    # head-to-head lift vs naive
    lift = lift_vs_naive(per_trade)
    if not lift.empty:
        lift_path = OUT_DIR / "lift_vs_naive.csv"
        lift.to_csv(lift_path, index=False)
        print(f"[out] {lift_path}")

    # hybrid policies
    hybrid_pt = synthesize_hybrid_policies(per_trade)
    if not hybrid_pt.empty:
        hybrid_summary = variant_summary(hybrid_pt)
        hyb_path = OUT_DIR / "hybrid_policies.csv"
        hybrid_summary.to_csv(hyb_path, index=False)
        print(f"[out] {hyb_path}")
    else:
        hybrid_summary = pd.DataFrame()

    # xlsx
    try:
        with pd.ExcelWriter(OUT_DIR / "variants_summary.xlsx", engine="openpyxl") as xw:
            summary.to_excel(xw, sheet_name="variants", index=False)
            if not lift.empty:
                lift.to_excel(xw, sheet_name="lift_vs_naive", index=False)
            if not hybrid_summary.empty:
                hybrid_summary.to_excel(xw, sheet_name="hybrid_policies", index=False)
    except Exception as e:
        print(f"[warn] xlsx write skipped: {e}")

    # ---- console teaser -----------------------------------------------------
    show = ["variant", "regime", "prob_bucket", "n_taken", "trigger_pct",
            "mean_net_pct", "win_pct", "port_ann_sharpe", "port_max_dd_pct"]
    print("\n=== variants_comparison (sorted by Sharpe per bucket) ===")
    s = summary.copy()
    s = s.sort_values(["regime", "prob_bucket", "port_ann_sharpe"],
                      ascending=[True, True, False])
    print(s[show].to_string(index=False, float_format=lambda x: f"{x:7.2f}" if pd.notna(x) else "    nan"))

    if not hybrid_summary.empty:
        print("\n=== hybrid_policies (head-to-head) ===")
        print(hybrid_summary[show].to_string(index=False,
              float_format=lambda x: f"{x:7.2f}" if pd.notna(x) else "    nan"))


if __name__ == "__main__":
    main()
