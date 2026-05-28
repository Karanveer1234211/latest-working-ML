"""
entry_variants_v4.py
====================
Full-rerun comparison: v2 cheap-entries + v3 feature-aligned + v4 late-day
quality screen (signal-day intraday) + two-stage hybrid policies.

WHAT'S NEW IN V4
----------------
1. Late-day signal-day intraday quality features (from intraday cache):
     sig_close_strength       (close - day_low) / day_range
     sig_last_hour_ret_pct    (close / price_at_14:30 - 1) * 100
     sig_above_vwap_at_close  bool(close > intraday VWAP)
     sig_late_volume_ratio    last_hour_vol / first_hour_vol
     sig_tail_strength        (close - last_hour_low) / day_range
     sig_day_range_pct        day_range / close

2. New variants (signal-day quality screens, all enter at T+1 open):
     STRONG_CLOSE_NAIVE       sig_close_strength > 0.70
     LATE_HOUR_GREEN_NAIVE    sig_last_hour_ret_pct > 0
     ABOVE_VWAP_CLOSE_NAIVE   sig_above_vwap_at_close
     LATE_VOL_SURGE_NAIVE     sig_late_volume_ratio > 1.0
     TAIL_STRENGTH_NAIVE      sig_tail_strength > 0.5
     LATE_QUALITY_3of5        3 of 5 above pass
     LATE_QUALITY_4of5        4 of 5 above pass

3. Two-stage hybrids (signal-day pre-screen + T+1 morning confirmation):
     LATEQUAL_PLUS_DIST20H        late_quality_3of5 AND DIST20H_BREAK
     LATEQUAL_PLUS_CONF2OF4       late_quality_3of5 AND CONFLUENCE_2of4
     LATEQUAL_PLUS_DIST52WH       late_quality_3of5 AND DIST52WH_BREAK

4. Data fixes:
     - reads bar volumes from intraday cache so DVOL_SURGE_30m actually fires
     - recomputes prob_bucket from probability if missing/NaN

5. Caches enriched paths to disk so subsequent runs skip the slow intraday
   reads. First run takes ~10-20 min on 30k signals; reruns take seconds.

USAGE
-----
    python entry_variants_v4.py
    # First run does the intraday enrichment (slow, one-time).
    # Output: BASE_DIR/entry_variants_v4/

    python entry_variants_v4.py --refresh       # rebuild enriched cache
    python entry_variants_v4.py --limit 2000    # test on first 2000 signals

OUTPUTS
-------
  enriched_paths_v4.parquet         signals + panel + late-day features
  per_trade_v4.parquet              one row per (signal x variant)
  variants_comparison_v4.csv        headline metrics by (variant x regime x bucket)
  hybrid_policies_v4.csv            hybrid policies head-to-head
  late_day_diagnostics.csv          how the late-day features distribute
  variants_summary_v4.xlsx          all of the above as tabs
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
# CONFIG
# =============================================================================

BASE_DIR = Path(
    os.environ.get("EVV4_BASE_DIR", r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
)
DEFAULT_PATHS = BASE_DIR / "orb_machine_results_v2" / "extracted_paths_v2.parquet"
DEFAULT_PANEL = BASE_DIR / "panel_cache.parquet"
DEFAULT_INTRADAY = Path(
    os.environ.get("EVV4_INTRADAY_DIR", r"C:\Users\karanvsi\Desktop\Pycharm\Cache\intraday_5min")
)
DEFAULT_OUT = BASE_DIR / "entry_variants_v4"

IST = "Asia/Kolkata"

SIGNAL_PROB_MIN = 0.65
PROB_BUCKETS = [(0.65, 0.70), (0.70, 0.75), (0.75, 0.85), (0.85, 1.01)]

HOLDING_DAYS = 5
COST_BPS_ROUND_TRIP = 25
SLIPPAGE_BPS_PER_SIDE = 5
TOTAL_COST_PCT = (COST_BPS_ROUND_TRIP + 2 * SLIPPAGE_BPS_PER_SIDE) / 100.0  # ~0.35%

TP_LEVELS_PCT = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
SL_LEVELS_PCT = [-1.0, -2.0, -3.0, -5.0]

# bar windows
FIRST_15M_BARS = 3
FIRST_30M_BARS = 6
FIRST_HOUR_BARS = 12

# v3 feature thresholds
RANGE_EXPAND_FRAC = 0.5
DVOL_SURGE_FRAC_OF_DAY = 0.25
GAPDOWN_REJECT_PCT = -0.5

# v4 late-day quality thresholds
CLOSE_STRENGTH_THRESH = 0.70
TAIL_STRENGTH_THRESH = 0.50
LATE_VOL_RATIO_THRESH = 1.00
LATE_QUALITY_K_3OF5 = 3
LATE_QUALITY_K_4OF5 = 4

# =============================================================================
# VARIANT CATALOG
# =============================================================================

VARIANTS: Dict[str, Dict] = {
    # ============== Baselines ==============
    "NAIVE_T1_OPEN": {
        "label": "Naive T+1 open entry, no filter",
        "engine": "alt", "confirm": "naive_open",
    },
    "ORB_15_held_anyday_static": {
        "label": "Reference: 15-min ORB, held EOD, T+1..T+5",
        "engine": "orb",
    },

    # ============== v2 Cheap-entry filters ==============
    "GREEN_15m_t1": {
        "label": "Green 09:15-09:30 candle, enter at 09:30 close",
        "engine": "alt", "confirm": "first_bar_green",
    },
    "ABOVE_PREVCLOSE_OPEN": {
        "label": "Enter at T+1 open if open >= prev_close",
        "engine": "alt", "confirm": "open_above_prev",
    },
    "VWAP_RECLAIM_30m": {
        "label": "Enter on first 5m close >= intraday VWAP by 09:45",
        "engine": "alt", "confirm": "vwap_reclaim",
    },
    "PREVCLOSE_HOLD_EOD": {
        "label": "Enter at T+1 close if T+1 close >= prev_close",
        "engine": "alt", "confirm": "prevclose_hold_eod",
    },
    "LIMIT_AT_PREVCLOSE": {
        "label": "Buy limit at prev_close; reject if T+1 gaps down >0.5%",
        "engine": "alt", "confirm": "limit_at_prevclose",
    },

    # ============== v3 Feature-aligned (model's top-gain features) ==============
    "DIST52WH_BREAK": {
        "label": "Enter when T+1 5-min close > prev 252-day high",
        "engine": "feat", "confirm": "dist52wh_break",
    },
    "DIST20H_BREAK": {
        "label": "Enter when T+1 5-min close > prev 20-day high",
        "engine": "feat", "confirm": "dist20h_break",
    },
    "RANGE_EXPAND_15m": {
        "label": f"Enter at 09:30 if first-15m range >= "
                 f"{RANGE_EXPAND_FRAC} * prev day range",
        "engine": "feat", "confirm": "range_expand_15m",
    },
    "DVOL_SURGE_30m": {
        "label": f"Enter at 09:45 if 09:15-09:45 dvol >= "
                 f"{DVOL_SURGE_FRAC_OF_DAY} * prev_avg_dvol20",
        "engine": "feat", "confirm": "dvol_surge_30m",
    },
    "CONFLUENCE_2of4": {
        "label": "2 of {DIST52WH, DIST20H, RANGE_EXPAND, DVOL_SURGE} by 09:45",
        "engine": "conf", "confirm": "confluence", "k": 2,
    },
    "CONFLUENCE_3of4": {
        "label": "3 of {DIST52WH, DIST20H, RANGE_EXPAND, DVOL_SURGE} by 09:45",
        "engine": "conf", "confirm": "confluence", "k": 3,
    },

    # ============== v4 Late-day quality screens (signal-day intraday) ==============
    "STRONG_CLOSE_NAIVE": {
        "label": f"Enter at T+1 open if signal-day close_strength > {CLOSE_STRENGTH_THRESH}",
        "engine": "lateq", "confirm": "strong_close",
    },
    "LATE_HOUR_GREEN_NAIVE": {
        "label": "Enter at T+1 open if signal-day last hour return > 0",
        "engine": "lateq", "confirm": "late_hour_green",
    },
    "ABOVE_VWAP_CLOSE_NAIVE": {
        "label": "Enter at T+1 open if signal-day close > intraday VWAP",
        "engine": "lateq", "confirm": "above_vwap_close",
    },
    "LATE_VOL_SURGE_NAIVE": {
        "label": f"Enter at T+1 open if last_hour_vol > {LATE_VOL_RATIO_THRESH} * first_hour_vol",
        "engine": "lateq", "confirm": "late_vol_surge",
    },
    "TAIL_STRENGTH_NAIVE": {
        "label": f"Enter at T+1 open if (close - last_hour_low)/day_range > {TAIL_STRENGTH_THRESH}",
        "engine": "lateq", "confirm": "tail_strength",
    },
    "LATE_QUALITY_3of5": {
        "label": "Enter at T+1 open if 3 of 5 late-day quality checks pass",
        "engine": "lateq", "confirm": "late_quality", "k": LATE_QUALITY_K_3OF5,
    },
    "LATE_QUALITY_4of5": {
        "label": "Enter at T+1 open if 4 of 5 late-day quality checks pass",
        "engine": "lateq", "confirm": "late_quality", "k": LATE_QUALITY_K_4OF5,
    },

    # ============== v4 Two-stage hybrids (late-day + morning) ==============
    "LATEQUAL_PLUS_DIST20H": {
        "label": "Stage 1: late_quality_3of5; Stage 2: DIST20H_BREAK",
        "engine": "twostage", "confirm": "twostage",
        "stage1": "late_quality", "stage1_k": LATE_QUALITY_K_3OF5,
        "stage2": "dist20h_break",
    },
    "LATEQUAL_PLUS_CONF2OF4": {
        "label": "Stage 1: late_quality_3of5; Stage 2: CONFLUENCE_2of4",
        "engine": "twostage", "confirm": "twostage",
        "stage1": "late_quality", "stage1_k": LATE_QUALITY_K_3OF5,
        "stage2": "confluence", "stage2_k": 2,
    },
    "LATEQUAL_PLUS_DIST52WH": {
        "label": "Stage 1: late_quality_3of5; Stage 2: DIST52WH_BREAK (bull only)",
        "engine": "twostage", "confirm": "twostage",
        "stage1": "late_quality", "stage1_k": LATE_QUALITY_K_3OF5,
        "stage2": "dist52wh_break",
    },
}

# =============================================================================
# DATA  --  prob_bucket fix + panel join + intraday enrichment
# =============================================================================

def ensure_prob_bucket(paths: pd.DataFrame) -> pd.DataFrame:
    """v3 lost prob_bucket in the merge. Recompute if missing or all-NaN."""
    needs = ("prob_bucket" not in paths.columns) or paths["prob_bucket"].isna().all()
    if not needs:
        return paths
    paths = paths.copy()
    bins = [lo for lo, _ in PROB_BUCKETS] + [PROB_BUCKETS[-1][1]]
    labels = [f"[{lo:.2f},{hi:.2f})" for lo, hi in PROB_BUCKETS]
    paths["prob_bucket"] = pd.cut(
        paths["probability"], bins=bins, right=False, labels=labels
    ).astype(str)
    print(f"[fix] recomputed prob_bucket from probability "
          f"({paths['prob_bucket'].nunique()} buckets)")
    return paths


def enrich_with_panel(paths: pd.DataFrame, panel_path: Path) -> pd.DataFrame:
    """Same as v3: attach prev_high_20d, prev_high_252d, prev_avg_dvol20,
    prev_range_pct, prev_atr14."""
    print(f"[panel] reading: {panel_path}")
    pn = pd.read_parquet(panel_path)
    pn["timestamp"] = pd.to_datetime(pn["timestamp"])
    if pn["timestamp"].dt.tz is None:
        pn["timestamp"] = pn["timestamp"].dt.tz_localize(IST)
    pn = pn.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    g = pn.groupby("symbol", group_keys=False)
    pn["prev_high_20d"]  = g["high"].apply(
        lambda s: s.shift(1).rolling(20, min_periods=10).max())
    pn["prev_high_252d"] = g["high"].apply(
        lambda s: s.shift(1).rolling(252, min_periods=100).max())

    if "D_dollar_vol" in pn.columns:
        pn["prev_avg_dvol20"] = g["D_dollar_vol"].apply(
            lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    elif "volume" in pn.columns and "close" in pn.columns:
        pn["dollar_vol"] = pn["close"] * pn["volume"]
        pn["prev_avg_dvol20"] = g["dollar_vol"].apply(
            lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    else:
        pn["prev_avg_dvol20"] = np.nan

    pn["prev_range_pct"] = pn.get("D_range_pct",
                                   (pn["high"] - pn["low"]) / pn["close"])
    pn["prev_atr14"] = pn.get("D_atr14", np.nan)

    keep = ["symbol", "timestamp", "prev_high_20d", "prev_high_252d",
            "prev_range_pct", "prev_avg_dvol20", "prev_atr14"]
    keep = [c for c in keep if c in pn.columns]
    pn_join = pn[keep].rename(columns={"timestamp": "signal_date"})

    paths = paths.copy()
    paths["signal_date"] = pd.to_datetime(paths["signal_date"])
    if paths["signal_date"].dt.tz is None:
        paths["signal_date"] = paths["signal_date"].dt.tz_localize(IST)

    n_before = len(paths)
    out = paths.merge(pn_join, on=["symbol", "signal_date"], how="left")
    n_levels = int(out["prev_high_20d"].notna().sum())
    print(f"[panel] joined {n_levels:,}/{n_before:,} signals "
          f"({100*n_levels/max(n_before,1):.1f}%)")
    return out


def _filter_market_hours(b: pd.DataFrame) -> pd.DataFrame:
    """09:15 <= ts <= 15:30."""
    ts = b["timestamp"]
    return b[((ts.dt.hour > 9) | ((ts.dt.hour == 9) & (ts.dt.minute >= 15))) &
            ((ts.dt.hour < 15) | ((ts.dt.hour == 15) & (ts.dt.minute <= 30)))]


def _compute_late_day_features(sig_bars: pd.DataFrame) -> Dict[str, float]:
    """7 quality features from one day's intraday bars (5-min)."""
    if len(sig_bars) < 12:
        return {}
    sig_bars = sig_bars.sort_values("timestamp").reset_index(drop=True)

    day_open  = float(sig_bars.iloc[0]["open"])
    day_close = float(sig_bars.iloc[-1]["close"])
    day_high  = float(sig_bars["high"].max())
    day_low   = float(sig_bars["low"].min())
    day_range = day_high - day_low
    if day_range <= 0 or day_close <= 0:
        return {}

    out: Dict[str, float] = {}
    out["sig_close_strength"] = (day_close - day_low) / day_range

    if len(sig_bars) >= 12:
        price_14_30 = float(sig_bars.iloc[-12]["open"])
        if price_14_30 > 0:
            out["sig_last_hour_ret_pct"] = (day_close / price_14_30 - 1) * 100

    typical = (sig_bars["high"] + sig_bars["low"] + sig_bars["close"]) / 3.0
    if "volume" in sig_bars.columns:
        vol = pd.to_numeric(sig_bars["volume"], errors="coerce").fillna(0.0)
    else:
        vol = pd.Series(np.ones(len(sig_bars)))
    if vol.sum() > 0:
        vwap = float((typical * vol).sum() / vol.sum())
    else:
        vwap = float(typical.mean())
    out["sig_above_vwap_at_close"] = float(day_close > vwap)  # store as 0/1

    if "volume" in sig_bars.columns and len(sig_bars) >= 24:
        last_hour_vol  = float(sig_bars.tail(12)["volume"].sum())
        first_hour_vol = float(sig_bars.head(12)["volume"].sum())
        if first_hour_vol > 0:
            out["sig_late_volume_ratio"] = last_hour_vol / first_hour_vol

    if len(sig_bars) >= 12:
        last_hour = sig_bars.tail(12)
        last_hour_low = float(last_hour["low"].min())
        out["sig_tail_strength"] = (day_close - last_hour_low) / day_range

    out["sig_day_range_pct"] = day_range / day_close
    out["sig_close"]    = day_close
    out["sig_vwap"]     = vwap
    out["sig_day_open"] = day_open
    return out


def _compute_t1_volume_metrics(t1_bars: pd.DataFrame) -> Dict[str, float]:
    """First-30-min cumulative dollar volume on T+1, for DVOL_SURGE_30m."""
    out: Dict[str, float] = {"t1_first_30m_dvol": np.nan}
    if t1_bars.empty or len(t1_bars) < FIRST_30M_BARS:
        return out
    t1_bars = t1_bars.sort_values("timestamp").reset_index(drop=True)
    head = t1_bars.head(FIRST_30M_BARS)
    if "volume" in head.columns:
        vol = pd.to_numeric(head["volume"], errors="coerce").fillna(0.0)
        out["t1_first_30m_dvol"] = float((head["close"] * vol).sum())
    return out


def enrich_with_intraday(paths: pd.DataFrame, intraday_dir: Path) -> pd.DataFrame:
    """
    For each signal, read the symbol's 5-min cache once, compute late-day
    features for signal_date and T+1 first-30-min volume.

    Reads each symbol's parquet ONCE and processes all of its signals together
    for performance.
    """
    print(f"[intraday] reading from: {intraday_dir}")
    paths = paths.copy()
    new_cols = ["sig_close_strength", "sig_last_hour_ret_pct",
                "sig_above_vwap_at_close", "sig_late_volume_ratio",
                "sig_tail_strength", "sig_day_range_pct",
                "sig_close", "sig_vwap", "sig_day_open",
                "t1_first_30m_dvol"]
    for c in new_cols:
        if c not in paths.columns:
            paths[c] = np.nan

    paths["signal_date"] = pd.to_datetime(paths["signal_date"])
    if paths["signal_date"].dt.tz is None:
        paths["signal_date"] = paths["signal_date"].dt.tz_localize(IST)

    symbols = paths["symbol"].unique().tolist()
    n_sym = len(symbols)
    print(f"[intraday] {len(paths):,} signals across {n_sym:,} symbols")

    n_done_sym = 0
    n_done_sigs = 0
    n_missing_files = 0
    n_with_late_day = 0
    n_with_t1_vol = 0

    for sym in symbols:
        n_done_sym += 1
        if n_done_sym % 250 == 0:
            print(f"   [intraday] {n_done_sym}/{n_sym} symbols processed, "
                  f"{n_done_sigs:,} signals enriched")

        fp = intraday_dir / f"{sym}.parquet"
        if not fp.exists():
            n_missing_files += 1
            continue
        try:
            bars = pd.read_parquet(fp)
        except Exception:
            n_missing_files += 1
            continue
        if bars.empty:
            continue

        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
        if bars["timestamp"].dt.tz is None:
            bars["timestamp"] = bars["timestamp"].dt.tz_localize(IST)
        bars = bars.sort_values("timestamp").reset_index(drop=True)
        bars = _filter_market_hours(bars)
        if bars.empty:
            continue

        # group by trading day for fast lookups
        bars_by_day = {d: g.reset_index(drop=True)
                       for d, g in bars.groupby(bars["timestamp"].dt.normalize())}

        sym_signals = paths[paths["symbol"] == sym]
        for idx, row in sym_signals.iterrows():
            sd = pd.Timestamp(row["signal_date"]).normalize()

            sig_bars = bars_by_day.get(sd, None)

            if sig_bars is not None and not sig_bars.empty:
                feats = _compute_late_day_features(sig_bars)
                for k, v in feats.items():
                    paths.at[idx, k] = v
                if "sig_close_strength" in feats:
                    n_with_late_day += 1

            after = sorted([d for d in bars_by_day.keys() if d > sd])
            if after:
                t1_bars = bars_by_day[after[0]]
                tv = _compute_t1_volume_metrics(t1_bars)
                for k, v in tv.items():
                    paths.at[idx, k] = v
                if pd.notna(tv["t1_first_30m_dvol"]):
                    n_with_t1_vol += 1

            n_done_sigs += 1

    print(f"[intraday] symbols missing parquet: {n_missing_files:,}")
    print(f"[intraday] signals with late-day features: {n_with_late_day:,}")
    print(f"[intraday] signals with T+1 30-min volume:  {n_with_t1_vol:,}")
    return paths


# =============================================================================
# OCO + FORWARD WALK
# =============================================================================

@dataclass
class TradeOutcome:
    triggered: bool = False
    entry_idx: int = -1
    entry_price: float = np.nan
    entry_day_offset: int = -1
    ref_high: float = np.nan
    ref_low: float = np.nan
    mae_pct: float = np.nan
    mfe_pct: float = np.nan
    fwd_return_to_t5_close_pct: float = np.nan
    oco: Optional[Dict] = None
    confirm_count: int = 0


def _walk_oco(bar_highs, bar_lows, bar_closes, bar_days,
              entry_idx, entry_price, orl, orh, holding_days):
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
    fwd = (bar_closes[last_idx] / entry_price - 1) * 100

    oco = {}
    sl_special = [("range_low", orl),
                  ("two_range", entry_price - 2 * (orh - orl))]
    sl_pct_levels = [(f"{lvl:.1f}", entry_price * (1 + lvl / 100))
                     for lvl in SL_LEVELS_PCT]
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
                ret = fwd
            oco[(tp_pct, sl_name)] = {
                "hit_tp": hit_tp, "hit_sl": hit_sl,
                "ret_pct": ret, "bars_to_exit": exit_idx - entry_idx,
            }
    return oco, raw_mae, raw_mfe, fwd


def _ref_levels(entry_price):
    return entry_price * 1.01, entry_price * 0.99


def _day1_idx(bar_days):
    return np.where(bar_days == 0)[0]


# =============================================================================
# ATOMIC INTRADAY CONFIRMATIONS  (T+1 morning)
# =============================================================================

def _check_dist52wh_break(d1_idx, bar_closes, prev_high_252d):
    if pd.isna(prev_high_252d):
        return -1, np.nan
    for i in d1_idx:
        if bar_closes[i] > prev_high_252d:
            return int(i), float(bar_closes[i])
    return -1, np.nan


def _check_dist20h_break(d1_idx, bar_closes, prev_high_20d):
    if pd.isna(prev_high_20d):
        return -1, np.nan
    for i in d1_idx:
        if bar_closes[i] > prev_high_20d:
            return int(i), float(bar_closes[i])
    return -1, np.nan


def _check_range_expand_15m(d1_idx, bar_highs, bar_lows, bar_closes,
                             prev_close, prev_range_pct):
    if pd.isna(prev_range_pct) or pd.isna(prev_close) or len(d1_idx) < FIRST_15M_BARS:
        return -1, np.nan
    bars = d1_idx[:FIRST_15M_BARS]
    rng = float(np.max(bar_highs[bars]) - np.min(bar_lows[bars]))
    threshold = RANGE_EXPAND_FRAC * float(prev_range_pct) * float(prev_close)
    if rng < threshold:
        return -1, np.nan
    i = int(bars[-1])
    return i, float(bar_closes[i])


def _check_dvol_surge_30m_from_scalar(d1_idx, bar_closes,
                                       t1_first_30m_dvol, prev_avg_dvol20):
    """v4: T+1 volume comes from a scalar pre-computed during enrichment."""
    if pd.isna(prev_avg_dvol20) or pd.isna(t1_first_30m_dvol):
        return -1, np.nan
    if len(d1_idx) < FIRST_30M_BARS:
        return -1, np.nan
    threshold = DVOL_SURGE_FRAC_OF_DAY * float(prev_avg_dvol20)
    if float(t1_first_30m_dvol) < threshold:
        return -1, np.nan
    i = int(d1_idx[FIRST_30M_BARS - 1])
    return i, float(bar_closes[i])


# =============================================================================
# v4 LATE-DAY QUALITY CHECKS  (signal-day intraday)
# =============================================================================

def _passes_late_day_check(check_name: str, sig: pd.Series) -> Optional[bool]:
    """Returns True/False if the check is computable, None if data missing."""
    cs = sig.get("sig_close_strength")
    lh = sig.get("sig_last_hour_ret_pct")
    av = sig.get("sig_above_vwap_at_close")
    lv = sig.get("sig_late_volume_ratio")
    ts = sig.get("sig_tail_strength")

    if check_name == "strong_close":
        if pd.isna(cs): return None
        return float(cs) > CLOSE_STRENGTH_THRESH
    if check_name == "late_hour_green":
        if pd.isna(lh): return None
        return float(lh) > 0
    if check_name == "above_vwap_close":
        if pd.isna(av): return None
        return bool(av >= 0.5)
    if check_name == "late_vol_surge":
        if pd.isna(lv): return None
        return float(lv) > LATE_VOL_RATIO_THRESH
    if check_name == "tail_strength":
        if pd.isna(ts): return None
        return float(ts) > TAIL_STRENGTH_THRESH
    return None


def _late_quality_count(sig: pd.Series) -> Tuple[int, int]:
    """Returns (n_pass, n_evaluable). Evaluable = check had data."""
    checks = ["strong_close", "late_hour_green", "above_vwap_close",
              "late_vol_surge", "tail_strength"]
    n_pass = 0
    n_eval = 0
    for c in checks:
        r = _passes_late_day_check(c, sig)
        if r is None:
            continue
        n_eval += 1
        if r:
            n_pass += 1
    return n_pass, n_eval


# =============================================================================
# DISPATCHERS
# =============================================================================

def _find_entry_naive(bar_opens, bar_days):
    d1 = _day1_idx(bar_days)
    if len(d1) == 0:
        return -1, np.nan, np.nan, np.nan
    i = int(d1[0])
    px = float(bar_opens[i])
    rh, rl = _ref_levels(px)
    return i, px, rh, rl


def _find_entry_orb_held(bar_highs, bar_lows, bar_closes, bar_days, sig):
    """ORB_15_held_anyday_static. Static T+1 ORH/ORL, held EOD."""
    orh = sig.get("t1_orh_15min")
    orl = sig.get("t1_orl_15min")
    if pd.isna(orh) or pd.isna(orl) or orh <= orl:
        return -1, np.nan, np.nan, np.nan
    n = len(bar_closes)
    for i in range(n):
        d = int(bar_days[i])
        if d < 0 or d >= HOLDING_DAYS:
            continue
        if d == 0:
            same_day_before = int(np.sum(bar_days[:i] == 0))
            if same_day_before < FIRST_15M_BARS:
                continue
        if bar_closes[i] > orh:
            same_day = np.where(bar_days == d)[0]
            if len(same_day) and bar_closes[same_day[-1]] > orh:
                return i, float(bar_closes[i]), float(orh), float(orl)
    return -1, np.nan, np.nan, np.nan


def _find_entry_alt(bar_opens, bar_highs, bar_lows, bar_closes, bar_days,
                    variant, sig):
    """v2 cheap-entry filters."""
    confirm = variant["confirm"]
    d1 = _day1_idx(bar_days)
    if len(d1) == 0:
        return -1, np.nan, np.nan, np.nan
    open_t1 = float(bar_opens[d1[0]])
    prev_close = float(sig.get("prev_close", np.nan))

    if confirm == "naive_open":
        i = int(d1[0])
        return i, open_t1, *_ref_levels(open_t1)

    if confirm == "open_above_prev":
        if pd.isna(prev_close) or open_t1 < prev_close:
            return -1, np.nan, np.nan, np.nan
        i = int(d1[0])
        return i, open_t1, *_ref_levels(open_t1)

    if confirm == "first_bar_green":
        if len(d1) < FIRST_15M_BARS:
            return -1, np.nan, np.nan, np.nan
        bars = d1[:FIRST_15M_BARS]
        bo = float(bar_opens[bars[0]])
        bc = float(bar_closes[bars[-1]])
        if bc <= bo:
            return -1, np.nan, np.nan, np.nan
        return int(bars[-1]), bc, *_ref_levels(bc)

    if confirm == "vwap_reclaim":
        deadline = min(FIRST_30M_BARS, len(d1))
        if deadline < 2:
            return -1, np.nan, np.nan, np.nan
        d1_h = bar_highs[d1[:deadline]]
        d1_l = bar_lows[d1[:deadline]]
        d1_c = bar_closes[d1[:deadline]]
        for k in range(1, deadline):
            tp = (d1_h[:k+1] + d1_l[:k+1] + d1_c[:k+1]) / 3.0
            vwap = float(np.mean(tp))
            if d1_c[k] >= vwap:
                px = float(d1_c[k])
                return int(d1[k]), px, *_ref_levels(px)
        return -1, np.nan, np.nan, np.nan

    if confirm == "prevclose_hold_eod":
        if pd.isna(prev_close):
            return -1, np.nan, np.nan, np.nan
        last = int(d1[-1])
        c = float(bar_closes[last])
        if c < prev_close:
            return -1, np.nan, np.nan, np.nan
        return last, c, *_ref_levels(c)

    if confirm == "limit_at_prevclose":
        if pd.isna(prev_close):
            return -1, np.nan, np.nan, np.nan
        gap_pct = (open_t1 / prev_close - 1.0) * 100.0
        if gap_pct < GAPDOWN_REJECT_PCT:
            return -1, np.nan, np.nan, np.nan
        if open_t1 <= prev_close:
            i = int(d1[0])
            return i, open_t1, *_ref_levels(open_t1)
        for i in d1:
            if float(bar_lows[i]) <= prev_close:
                return int(i), prev_close, *_ref_levels(prev_close)
        return -1, np.nan, np.nan, np.nan

    return -1, np.nan, np.nan, np.nan


def _find_entry_feat(bar_opens, bar_highs, bar_lows, bar_closes, bar_days,
                     variant, sig):
    """v3 feature-aligned single-feature variants."""
    confirm = variant["confirm"]
    d1 = _day1_idx(bar_days)
    if len(d1) == 0:
        return -1, np.nan, np.nan, np.nan

    open_t1 = float(bar_opens[d1[0]])
    prev_close = float(sig.get("prev_close", np.nan))
    if not pd.isna(prev_close):
        gap_pct = (open_t1 / prev_close - 1.0) * 100.0
        if gap_pct < GAPDOWN_REJECT_PCT:
            return -1, np.nan, np.nan, np.nan

    if confirm == "dist52wh_break":
        i, px = _check_dist52wh_break(d1, bar_closes, sig.get("prev_high_252d"))
    elif confirm == "dist20h_break":
        i, px = _check_dist20h_break(d1, bar_closes, sig.get("prev_high_20d"))
    elif confirm == "range_expand_15m":
        i, px = _check_range_expand_15m(d1, bar_highs, bar_lows, bar_closes,
                                         prev_close, sig.get("prev_range_pct"))
    elif confirm == "dvol_surge_30m":
        i, px = _check_dvol_surge_30m_from_scalar(
            d1, bar_closes, sig.get("t1_first_30m_dvol"),
            sig.get("prev_avg_dvol20"))
    else:
        return -1, np.nan, np.nan, np.nan

    if i < 0:
        return -1, np.nan, np.nan, np.nan
    rh, rl = _ref_levels(px)
    return i, px, rh, rl


def _find_entry_confluence(bar_opens, bar_highs, bar_lows, bar_closes, bar_days,
                           variant, sig):
    """v3 confluence (k of 4 atomic confirmations)."""
    k = int(variant.get("k", 2))
    d1 = _day1_idx(bar_days)
    if len(d1) == 0:
        return -1, np.nan, np.nan, np.nan, 0

    open_t1 = float(bar_opens[d1[0]])
    prev_close = float(sig.get("prev_close", np.nan))
    if not pd.isna(prev_close):
        gap_pct = (open_t1 / prev_close - 1.0) * 100.0
        if gap_pct < GAPDOWN_REJECT_PCT:
            return -1, np.nan, np.nan, np.nan, 0

    fires: Dict[str, int] = {}

    i52, _ = _check_dist52wh_break(d1, bar_closes, sig.get("prev_high_252d"))
    if i52 >= 0: fires["dist52wh"] = i52

    i20, _ = _check_dist20h_break(d1, bar_closes, sig.get("prev_high_20d"))
    if i20 >= 0: fires["dist20h"] = i20

    iR, _ = _check_range_expand_15m(d1, bar_highs, bar_lows, bar_closes,
                                     prev_close, sig.get("prev_range_pct"))
    if iR >= 0: fires["range"] = iR

    iD, _ = _check_dvol_surge_30m_from_scalar(
        d1, bar_closes, sig.get("t1_first_30m_dvol"),
        sig.get("prev_avg_dvol20"))
    if iD >= 0: fires["dvol"] = iD

    if len(fires) < k:
        return -1, np.nan, np.nan, np.nan, len(fires)
    sorted_idx = sorted(fires.values())
    entry_i = int(sorted_idx[k - 1])
    px = float(bar_closes[entry_i])
    rh, rl = _ref_levels(px)
    return entry_i, px, rh, rl, len(fires)


def _find_entry_lateq(bar_opens, bar_days, variant, sig):
    """v4 late-day quality screen variants. All enter at T+1 OPEN if pass."""
    confirm = variant["confirm"]

    if confirm == "late_quality":
        k = int(variant.get("k", LATE_QUALITY_K_3OF5))
        n_pass, n_eval = _late_quality_count(sig)
        if n_eval == 0 or n_pass < k:
            return -1, np.nan, np.nan, np.nan, n_pass
    else:
        r = _passes_late_day_check(confirm, sig)
        if r is None or r is False:
            return -1, np.nan, np.nan, np.nan, 0

    d1 = _day1_idx(bar_days)
    if len(d1) == 0:
        return -1, np.nan, np.nan, np.nan, 0
    i = int(d1[0])
    px = float(bar_opens[i])
    rh, rl = _ref_levels(px)
    confirm_count = 1 if confirm != "late_quality" else _late_quality_count(sig)[0]
    return i, px, rh, rl, confirm_count


def _find_entry_twostage(bar_opens, bar_highs, bar_lows, bar_closes, bar_days,
                         variant, sig):
    """v4 two-stage: stage1 = late-day pre-screen, stage2 = T+1 morning."""
    stage1 = variant.get("stage1", "late_quality")
    if stage1 == "late_quality":
        k1 = int(variant.get("stage1_k", LATE_QUALITY_K_3OF5))
        n_pass, n_eval = _late_quality_count(sig)
        if n_eval == 0 or n_pass < k1:
            return -1, np.nan, np.nan, np.nan, 0
    else:
        r = _passes_late_day_check(stage1, sig)
        if r is None or r is False:
            return -1, np.nan, np.nan, np.nan, 0

    stage2 = variant.get("stage2", "dist20h_break")
    proxy = dict(variant)
    proxy["confirm"] = stage2
    if stage2 == "confluence":
        proxy["k"] = int(variant.get("stage2_k", 2))
        i, px, rh, rl, cc = _find_entry_confluence(
            bar_opens, bar_highs, bar_lows, bar_closes, bar_days, proxy, sig)
        return i, px, rh, rl, cc
    else:
        i, px, rh, rl = _find_entry_feat(
            bar_opens, bar_highs, bar_lows, bar_closes, bar_days, proxy, sig)
        return i, px, rh, rl, 1 if i >= 0 else 0


# =============================================================================
# SIMULATE ONE SIGNAL x ONE VARIANT
# =============================================================================

def simulate_signal(sig: pd.Series, variant: Dict) -> TradeOutcome:
    out = TradeOutcome()
    bts = sig.get("bar_timestamps")
    if not isinstance(bts, (list, np.ndarray)) or len(bts) < 20:
        return out

    bar_opens  = np.asarray(sig["bar_opens"],  dtype=float)
    bar_highs  = np.asarray(sig["bar_highs"],  dtype=float)
    bar_lows   = np.asarray(sig["bar_lows"],   dtype=float)
    bar_closes = np.asarray(sig["bar_closes"], dtype=float)
    bar_days   = np.asarray(sig["bar_days"],   dtype=int)

    engine = variant.get("engine", "alt")
    confirm_count = 0

    if engine == "alt":
        e_idx, e_px, ref_hi, ref_lo = _find_entry_alt(
            bar_opens, bar_highs, bar_lows, bar_closes, bar_days, variant, sig)
    elif engine == "orb":
        e_idx, e_px, ref_hi, ref_lo = _find_entry_orb_held(
            bar_highs, bar_lows, bar_closes, bar_days, sig)
    elif engine == "feat":
        e_idx, e_px, ref_hi, ref_lo = _find_entry_feat(
            bar_opens, bar_highs, bar_lows, bar_closes, bar_days, variant, sig)
    elif engine == "conf":
        e_idx, e_px, ref_hi, ref_lo, confirm_count = _find_entry_confluence(
            bar_opens, bar_highs, bar_lows, bar_closes, bar_days, variant, sig)
    elif engine == "lateq":
        e_idx, e_px, ref_hi, ref_lo, confirm_count = _find_entry_lateq(
            bar_opens, bar_days, variant, sig)
    elif engine == "twostage":
        e_idx, e_px, ref_hi, ref_lo, confirm_count = _find_entry_twostage(
            bar_opens, bar_highs, bar_lows, bar_closes, bar_days, variant, sig)
    else:
        return out

    if e_idx < 0:
        out.confirm_count = confirm_count
        return out

    out.triggered = True
    out.entry_idx = e_idx
    out.entry_price = e_px
    out.entry_day_offset = int(bar_days[e_idx]) + 1
    out.ref_high, out.ref_low = ref_hi, ref_lo
    out.confirm_count = confirm_count

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
# RUN ALL
# =============================================================================

def run_all(paths_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    n = len(paths_df)
    print(f"[sim] {n:,} signals  x  {len(VARIANTS)} variants  =  "
          f"{n*len(VARIANTS):,} simulations")

    sigs_dicts = paths_df.to_dict(orient="records")

    for var_id, variant in VARIANTS.items():
        triggered = 0
        for i, sd in enumerate(sigs_dicts):
            if i and i % 5000 == 0:
                print(f"   {var_id}: {i}/{n}  triggered={triggered}")
            res = simulate_signal(pd.Series(sd), variant)
            base = {
                "variant": var_id,
                "engine": variant.get("engine", "alt"),
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
                "confirm_count": res.confirm_count,
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
        rate = 100.0 * triggered / max(n, 1)
        print(f"   {var_id}: trigger {triggered}/{n} = {rate:.1f}%")
    return pd.DataFrame(rows)


# =============================================================================
# AGGREGATIONS
# =============================================================================

def _portfolio_metrics(trades: pd.DataFrame, ret_col: str = "fwd_return_to_t5_close_pct") -> Dict:
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
    rows: List[Dict] = []
    grp_cols = ["variant", "regime", "prob_bucket"]
    for keys, sub in per_trade.groupby(grp_cols, dropna=False):
        var_id, regime, bucket = keys
        n_sig = len(sub)
        n_trig = int(sub["triggered"].sum())
        trig_rate = 100.0 * n_trig / max(n_sig, 1)
        taken = sub[sub["triggered"]].copy()
        if not taken.empty:
            mean_ret  = float(taken["fwd_return_to_t5_close_pct"].mean())
            med_ret   = float(taken["fwd_return_to_t5_close_pct"].median())
            win_rate  = 100.0 * float((taken["fwd_return_to_t5_close_pct"] > 0).mean())
            mean_mae  = float(taken["mae_pct"].mean())
            mean_mfe  = float(taken["mfe_pct"].mean())
            mean_slip = float(taken["slippage_from_signal_pct"].mean())
            net_ret   = mean_ret - TOTAL_COST_PCT
        else:
            mean_ret = med_ret = win_rate = mean_mae = mean_mfe = mean_slip = net_ret = np.nan
        port = _portfolio_metrics(taken)
        rows.append({
            "variant": var_id, "regime": regime, "prob_bucket": bucket,
            "n_signals": n_sig, "n_taken": n_trig, "trigger_pct": trig_rate,
            "mean_ret_pct": mean_ret, "median_ret_pct": med_ret, "win_pct": win_rate,
            "mean_net_pct": net_ret,
            "mean_mae_pct": mean_mae, "mean_mfe_pct": mean_mfe,
            "mean_slippage_from_signal_pct": mean_slip,
            **port,
        })
    return pd.DataFrame(rows)


def late_day_diagnostics(paths: pd.DataFrame) -> pd.DataFrame:
    """How do the late-day features distribute across regimes / buckets?"""
    cols = ["sig_close_strength", "sig_last_hour_ret_pct",
            "sig_above_vwap_at_close", "sig_late_volume_ratio",
            "sig_tail_strength", "sig_day_range_pct"]
    cols = [c for c in cols if c in paths.columns]
    if not cols:
        return pd.DataFrame()
    rows: List[Dict] = []
    for keys, sub in paths.groupby(["regime", "prob_bucket"], dropna=False):
        reg, b = keys
        rec = {"regime": reg, "prob_bucket": b, "n": len(sub)}
        for c in cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            rec[f"{c}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            rec[f"{c}_med"]  = float(vals.median()) if vals.notna().any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# HYBRID POLICIES
# =============================================================================

HYBRID_POLICIES = {
    "HYBRID_ORBHELD_THEN_NAIVE": {
        "[0.65,0.70)": "ORB_15_held_anyday_static",
        "[0.70,0.75)": "ORB_15_held_anyday_static",
        "[0.75,0.85)": "NAIVE_T1_OPEN",
        "[0.85,1.01)": "NAIVE_T1_OPEN",
    },
    "HYBRID_LATEQUAL_THEN_NAIVE": {
        "[0.65,0.70)": "LATE_QUALITY_3of5",
        "[0.70,0.75)": "LATE_QUALITY_3of5",
        "[0.75,0.85)": "NAIVE_T1_OPEN",
        "[0.85,1.01)": "NAIVE_T1_OPEN",
    },
    "HYBRID_TWOSTAGE_DIST20H": {
        "[0.65,0.70)": "LATEQUAL_PLUS_DIST20H",
        "[0.70,0.75)": "LATEQUAL_PLUS_DIST20H",
        "[0.75,0.85)": "NAIVE_T1_OPEN",
        "[0.85,1.01)": "NAIVE_T1_OPEN",
    },
    "HYBRID_TWOSTAGE_CONF": {
        "[0.65,0.70)": "LATEQUAL_PLUS_CONF2OF4",
        "[0.70,0.75)": "LATEQUAL_PLUS_CONF2OF4",
        "[0.75,0.85)": "NAIVE_T1_OPEN",
        "[0.85,1.01)": "NAIVE_T1_OPEN",
    },
    # Most aggressive: 4-of-5 quality on lowest bucket, drops to 3-of-5 above
    "HYBRID_QUALITY_TIERED": {
        "[0.65,0.70)": "LATE_QUALITY_4of5",
        "[0.70,0.75)": "LATE_QUALITY_3of5",
        "[0.75,0.85)": "NAIVE_T1_OPEN",
        "[0.85,1.01)": "NAIVE_T1_OPEN",
    },
}


def synthesize_hybrid_policies(per_trade: pd.DataFrame) -> pd.DataFrame:
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
    return pd.concat(pieces, ignore_index=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-file", type=str, default=str(DEFAULT_PATHS))
    ap.add_argument("--panel-file", type=str, default=str(DEFAULT_PANEL))
    ap.add_argument("--intraday-dir", type=str, default=str(DEFAULT_INTRADAY))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--prob-min", type=float, default=SIGNAL_PROB_MIN)
    ap.add_argument("--limit", type=int, default=0,
                    help="If > 0, run on only the first N signals (for testing)")
    ap.add_argument("--refresh", action="store_true",
                    help="Rebuild enriched parquet from scratch")
    args = ap.parse_args()

    paths_path = Path(args.paths_file)
    panel_path = Path(args.panel_file)
    intraday_dir = Path(args.intraday_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_cache = out_dir / "enriched_paths_v4.parquet"

    if enriched_cache.exists() and not args.refresh:
        print(f"[main] reusing enriched cache: {enriched_cache}")
        paths = pd.read_parquet(enriched_cache)
    else:
        if not paths_path.exists():
            raise SystemExit(f"FATAL: paths file not found: {paths_path}")
        if not panel_path.exists():
            raise SystemExit(f"FATAL: panel file not found: {panel_path}")
        if not intraday_dir.exists():
            raise SystemExit(f"FATAL: intraday dir not found: {intraday_dir}")

        print(f"[main] reading paths: {paths_path}")
        paths = pd.read_parquet(paths_path)
        paths = paths[paths["probability"] >= args.prob_min].reset_index(drop=True)
        print(f"[main] {len(paths):,} signals at prob >= {args.prob_min}")

        paths = ensure_prob_bucket(paths)
        paths = enrich_with_panel(paths, panel_path)
        paths = enrich_with_intraday(paths, intraday_dir)

        paths.to_parquet(enriched_cache, index=False)
        print(f"[main] cached enriched paths: {enriched_cache}")

    if args.limit > 0:
        paths = paths.head(args.limit).reset_index(drop=True)
        print(f"[main] LIMITED to first {len(paths):,} signals")

    # late-day feature distribution diagnostic
    diag = late_day_diagnostics(paths)
    if not diag.empty:
        diag.to_csv(out_dir / "late_day_diagnostics.csv", index=False)
        print(f"[out] {out_dir / 'late_day_diagnostics.csv'}")

    per_trade = run_all(paths)
    pt_path = out_dir / "per_trade_v4.parquet"
    per_trade.to_parquet(pt_path, index=False)
    print(f"[out] {pt_path}")

    summary = variant_summary(per_trade)
    summary.to_csv(out_dir / "variants_comparison_v4.csv", index=False)
    print(f"[out] {out_dir / 'variants_comparison_v4.csv'}")

    hybrid_pt = synthesize_hybrid_policies(per_trade)
    if not hybrid_pt.empty:
        hyb_summary = variant_summary(hybrid_pt)
        hyb_summary.to_csv(out_dir / "hybrid_policies_v4.csv", index=False)
        print(f"[out] {out_dir / 'hybrid_policies_v4.csv'}")
    else:
        hyb_summary = pd.DataFrame()

    try:
        with pd.ExcelWriter(out_dir / "variants_summary_v4.xlsx", engine="openpyxl") as xw:
            summary.to_excel(xw, sheet_name="variants", index=False)
            if not hyb_summary.empty:
                hyb_summary.to_excel(xw, sheet_name="hybrid", index=False)
            if not diag.empty:
                diag.to_excel(xw, sheet_name="late_day_diag", index=False)
    except Exception as e:
        print(f"[warn] xlsx write skipped: {e}")

    show = ["variant", "regime", "prob_bucket", "n_taken", "trigger_pct",
            "mean_net_pct", "win_pct", "port_ann_sharpe", "port_max_dd_pct"]
    print("\n=== variants_comparison_v4 (sorted per cell by Sharpe) ===")
    s = summary.sort_values(["regime", "prob_bucket", "port_ann_sharpe"],
                             ascending=[True, True, False])
    print(s[show].to_string(index=False, float_format=lambda x: f"{x:7.2f}" if pd.notna(x) else "    nan"))

    if not hyb_summary.empty:
        print("\n=== hybrid_policies_v4 ===")
        print(hyb_summary[show].to_string(index=False,
              float_format=lambda x: f"{x:7.2f}" if pd.notna(x) else "    nan"))


if __name__ == "__main__":
    main()
