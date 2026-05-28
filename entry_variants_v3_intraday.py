"""
entry_variants_v3_intraday.py
=============================
Feature-aligned intraday entry confirmations.

THESIS
------
The previous-day "filter" we applied (ORB / GREEN_15m / VWAP) used arbitrary
price-action criteria. Your own feature-pruning diagnostics say something
much more specific: the model's edge is concentrated in features that
literally measure breakouts (`D_dist_from_52wh`, `D_dist_from_20h`),
volatility expansion (`D_range_pct`), and volume confirmation
(`D_dollar_vol`). When T+1 morning prints a value that mechanically pushes
those features in the direction the model rewards, that is a *model-aligned*
intraday confirmation -- vastly more meaningful than ORB's "close > yesterday's
high" criterion.

The six variants below each map to a specific high-gain KEEP feature:

  Variant                  Feature             Rank  IC      AUC drop
  -----------------------  ------------------  ----  ------  --------
  DIST52WH_BREAK           D_dist_from_52wh    4     0.018   0.0111  ***
  DIST20H_BREAK            D_dist_from_20h     28    0.036   0.0004
  RANGE_EXPAND_15m         D_range_pct         6     0.041   0.0009
  DVOL_SURGE_30m           D_dollar_vol        12    0.017   0.0004
  CONFLUENCE_2of4          composite           --    --      --
  CONFLUENCE_3of4          composite           --    --      --

The DIST52WH_BREAK case is the highest-leverage single feature in your
production model -- bull_trend regime specialist with 10x the average
permutation drop. When intraday T+1 price clears the 252-day high, that
is the strongest signal-aligned confirmation available.

INPUTS
------
This module needs TWO parquet files:
  1. extracted_paths_v2.parquet  (from `ORB execution.py`)
  2. panel_cache.parquet         (from `New_model.py` -> the daily panel)

It joins the panel onto each signal row to grab daily features and rolling
levels (prev 20d high, prev 252d high, prev avg dollar volume, etc.).

USAGE
-----
    set EVV3_PATHS=C:\...\extracted_paths_v2.parquet
    set EVV3_PANEL=C:\...\panel_cache.parquet
    set EVV3_OUT=C:\...\entry_variants_v3
    python entry_variants_v3_intraday.py

    # or via CLI
    python entry_variants_v3_intraday.py \
        --paths-file ".../extracted_paths_v2.parquet" \
        --panel-file ".../panel_cache.parquet" \
        --out-dir    ".../entry_variants_v3"

OUTPUTS
-------
  per_trade_v3.parquet            one row per (signal x variant)
  variants_comparison_v3.csv      headline metrics
  hybrid_policies_v3.csv          feature-aligned hybrid head-to-head
  variants_summary_v3.xlsx        all of the above as tabs
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(
    os.environ.get("EVV3_BASE_DIR", r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
)
DEFAULT_PATHS = BASE_DIR / "orb_machine_results_v2" / "extracted_paths_v2.parquet"
DEFAULT_PANEL = BASE_DIR / "panel_cache.parquet"
DEFAULT_OUT = BASE_DIR / "entry_variants_v3"

IST = "Asia/Kolkata"

SIGNAL_PROB_MIN = 0.65
SIGNAL_REGIMES = ["bull_trend", "bear_trend"]
PROB_BUCKETS = [(0.65, 0.70), (0.70, 0.75), (0.75, 0.85), (0.85, 1.01)]

HOLDING_DAYS = 5

# costs
COST_BPS_ROUND_TRIP = 25
SLIPPAGE_BPS_PER_SIDE = 5
TOTAL_COST_PCT = (COST_BPS_ROUND_TRIP + 2 * SLIPPAGE_BPS_PER_SIDE) / 100.0  # ~0.35%

TP_LEVELS_PCT = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
SL_LEVELS_PCT = [-1.0, -2.0, -3.0, -5.0]

# ---- intraday windows -------------------------------------------------------
FIRST_15M_BARS = 3   # 09:15-09:30
FIRST_30M_BARS = 6   # 09:15-09:45
FIRST_HOUR_BARS = 12 # 09:15-10:15

# ---- feature thresholds (the actual triggers) -------------------------------
# RANGE_EXPAND fires when T+1's first 15-min realised range >=
#   `RANGE_EXPAND_FRAC` * (prev_day_range)
RANGE_EXPAND_FRAC = 0.5

# DVOL_SURGE fires when 09:15-09:45 dvol >=
#   `DVOL_SURGE_FRAC_OF_DAY` * prev_avg_dvol20
# A normal day distributes ~0.20 of full-day dvol in first 30 min, so
# 0.25 means ~25% above the normal first-30-min volume run-rate.
DVOL_SURGE_FRAC_OF_DAY = 0.25

# Gap-down rejection (always-on safety gate)
GAPDOWN_REJECT_PCT = -0.5

# =============================================================================
# VARIANT CATALOG
# =============================================================================

VARIANTS: Dict[str, Dict] = {
    # baselines for direct comparison
    "NAIVE_T1_OPEN": {
        "label": "Naive T+1 open entry, no filter",
        "engine": "alt", "confirm": "naive_open",
    },
    "ORB_15_held_anyday_static": {
        "label": "Reference: 15-min ORB, held EOD, T+1..T+5",
        "engine": "orb",
        "range_def": "15min", "watch_mode": "static_t1_5d", "confirm": "close_held_eod",
    },

    # feature-aligned single-feature variants
    "DIST52WH_BREAK": {
        "label": "Enter when T+1 5-min close > prev 252-day high",
        "engine": "feat", "confirm": "dist52wh_break",
        "needs": ["prev_high_252d"],
    },
    "DIST20H_BREAK": {
        "label": "Enter when T+1 5-min close > prev 20-day high",
        "engine": "feat", "confirm": "dist20h_break",
        "needs": ["prev_high_20d"],
    },
    "RANGE_EXPAND_15m": {
        "label": "Enter at 09:30 if first-15m range >= "
                 f"{RANGE_EXPAND_FRAC} * prev day range",
        "engine": "feat", "confirm": "range_expand_15m",
        "needs": ["prev_range_pct"],
    },
    "DVOL_SURGE_30m": {
        "label": "Enter at 09:45 if 09:15-09:45 dvol >= "
                 f"{DVOL_SURGE_FRAC_OF_DAY} * prev_avg_dvol20",
        "engine": "feat", "confirm": "dvol_surge_30m",
        "needs": ["prev_avg_dvol20"],
    },

    # confluence variants
    "CONFLUENCE_2of4": {
        "label": "2 of {DIST52WH, DIST20H, RANGE_EXPAND, DVOL_SURGE} by 09:45",
        "engine": "conf", "confirm": "confluence", "k": 2,
        "needs": ["prev_high_20d", "prev_high_252d", "prev_range_pct", "prev_avg_dvol20"],
    },
    "CONFLUENCE_3of4": {
        "label": "3 of {DIST52WH, DIST20H, RANGE_EXPAND, DVOL_SURGE} by 09:45",
        "engine": "conf", "confirm": "confluence", "k": 3,
        "needs": ["prev_high_20d", "prev_high_252d", "prev_range_pct", "prev_avg_dvol20"],
    },
}

# =============================================================================
# DATA: panel join (the v3-specific step)
# =============================================================================

def enrich_paths_with_panel_features(paths: pd.DataFrame, panel_path: Path) -> pd.DataFrame:
    """
    Join the daily panel onto each (symbol, signal_date) row to attach the
    columns we need to evaluate the new variants:

      - prev_high_20d   : rolling-20 max of high, shifted by 1
      - prev_high_252d  : rolling-252 max of high, shifted by 1
      - prev_range_pct  : (high - low) / close on signal day
      - prev_avg_dvol20 : rolling-20 mean of dollar volume, shifted by 1
      - prev_atr14      : ATR(14)  (already on panel as D_atr14)

    Volume column is also captured for an annualisation factor.
    """
    print(f"[panel] reading panel: {panel_path}")
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

    if "D_range_pct" in pn.columns:
        pn["prev_range_pct"] = pn["D_range_pct"]
    else:
        pn["prev_range_pct"] = (pn["high"] - pn["low"]) / pn["close"]

    if "D_atr14" in pn.columns:
        pn["prev_atr14"] = pn["D_atr14"]
    else:
        pn["prev_atr14"] = np.nan

    keep_cols = ["symbol", "timestamp", "prev_high_20d", "prev_high_252d",
                 "prev_range_pct", "prev_avg_dvol20", "prev_atr14"]
    keep_cols = [c for c in keep_cols if c in pn.columns]
    pn_join = pn[keep_cols].copy()

    paths = paths.copy()
    paths["signal_date"] = pd.to_datetime(paths["signal_date"])
    if paths["signal_date"].dt.tz is None:
        paths["signal_date"] = paths["signal_date"].dt.tz_localize(IST)
    pn_join = pn_join.rename(columns={"timestamp": "signal_date"})

    n_before = len(paths)
    out = paths.merge(pn_join, on=["symbol", "signal_date"], how="left")
    n_with_levels = int(out["prev_high_20d"].notna().sum())
    print(f"[panel] joined {n_with_levels:,}/{n_before:,} signals "
          f"with rolling levels ({100*n_with_levels/max(n_before,1):.1f}%)")
    return out

# =============================================================================
# CORE: forward walk + OCO ladder (copied from v2)
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
    oco: Optional[Dict[Tuple[float, str], Dict]] = None
    confirm_count: int = 0  # for confluence variants


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


# =============================================================================
# THE FOUR ATOMIC INTRADAY CONFIRMATIONS
# =============================================================================

def _check_dist52wh_break(d1_idx, bar_closes, prev_high_252d) -> Tuple[int, float]:
    """First 5-min bar on T+1 whose close > prev_high_252d. (-1, nan) if none."""
    if pd.isna(prev_high_252d):
        return -1, np.nan
    for i in d1_idx:
        if bar_closes[i] > prev_high_252d:
            return int(i), float(bar_closes[i])
    return -1, np.nan


def _check_dist20h_break(d1_idx, bar_closes, prev_high_20d) -> Tuple[int, float]:
    if pd.isna(prev_high_20d):
        return -1, np.nan
    for i in d1_idx:
        if bar_closes[i] > prev_high_20d:
            return int(i), float(bar_closes[i])
    return -1, np.nan


def _check_range_expand_15m(d1_idx, bar_highs, bar_lows, bar_closes,
                             prev_close, prev_range_pct) -> Tuple[int, float]:
    """
    By 09:30 close, has T+1's first-15m realised range exceeded
    RANGE_EXPAND_FRAC * (prev_range_pct * prev_close)?
    Entry index = the 09:30 bar.
    """
    if pd.isna(prev_range_pct) or pd.isna(prev_close) or len(d1_idx) < FIRST_15M_BARS:
        return -1, np.nan
    bars = d1_idx[:FIRST_15M_BARS]
    rng = float(np.max(bar_highs[bars]) - np.min(bar_lows[bars]))
    threshold = RANGE_EXPAND_FRAC * float(prev_range_pct) * float(prev_close)
    if rng < threshold:
        return -1, np.nan
    i = int(bars[-1])
    return i, float(bar_closes[i])


def _check_dvol_surge_30m(d1_idx, bar_volumes, bar_closes,
                          prev_avg_dvol20) -> Tuple[int, float]:
    """
    By 09:45 close, has cumulative dollar volume exceeded
    DVOL_SURGE_FRAC_OF_DAY * prev_avg_dvol20?
    """
    if pd.isna(prev_avg_dvol20) or bar_volumes is None or len(d1_idx) < FIRST_30M_BARS:
        return -1, np.nan
    bars = d1_idx[:FIRST_30M_BARS]
    dvol_cum = float(np.sum(np.asarray(bar_volumes)[bars] *
                            np.asarray(bar_closes)[bars]))
    threshold = DVOL_SURGE_FRAC_OF_DAY * float(prev_avg_dvol20)
    if dvol_cum < threshold:
        return -1, np.nan
    i = int(bars[-1])
    return i, float(bar_closes[i])


# =============================================================================
# DISPATCHERS
# =============================================================================

def _day1_idx(bar_days):
    return np.where(bar_days == 0)[0]


def _find_entry_feat(bar_opens, bar_highs, bar_lows, bar_closes, bar_volumes,
                     bar_days, variant, sig) -> Tuple[int, float, float, float]:
    """Single-feature intraday confirmations."""
    confirm = variant["confirm"]
    d1 = _day1_idx(bar_days)
    if len(d1) == 0:
        return -1, np.nan, np.nan, np.nan

    open_t1 = float(bar_opens[d1[0]])
    prev_close = float(sig.get("prev_close", np.nan))

    # always-on safety: skip on hard gap-downs
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
        i, px = _check_dvol_surge_30m(d1, bar_volumes, bar_closes,
                                       sig.get("prev_avg_dvol20"))
    else:
        return -1, np.nan, np.nan, np.nan

    if i < 0:
        return -1, np.nan, np.nan, np.nan
    rh, rl = _ref_levels(px)
    return i, px, rh, rl


def _find_entry_confluence(bar_opens, bar_highs, bar_lows, bar_closes, bar_volumes,
                           bar_days, variant, sig) -> Tuple[int, float, float, float, int]:
    """
    Confluence: enter at the EARLIEST bar by which `k` of the 4 atomic
    confirmations have triggered. Walks each atomic check, then takes the
    bar at which the k-th confirmation occurs.
    """
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

    # bar index at which each atomic confirmation triggers (or -1)
    fires = {}

    i52, _ = _check_dist52wh_break(d1, bar_closes, sig.get("prev_high_252d"))
    if i52 >= 0: fires["dist52wh"] = i52

    i20, _ = _check_dist20h_break(d1, bar_closes, sig.get("prev_high_20d"))
    if i20 >= 0: fires["dist20h"] = i20

    iR, _ = _check_range_expand_15m(d1, bar_highs, bar_lows, bar_closes,
                                     prev_close, sig.get("prev_range_pct"))
    if iR >= 0: fires["range"] = iR

    iD, _ = _check_dvol_surge_30m(d1, bar_volumes, bar_closes,
                                   sig.get("prev_avg_dvol20"))
    if iD >= 0: fires["dvol"] = iD

    if len(fires) < k:
        return -1, np.nan, np.nan, np.nan, len(fires)

    # entry at the bar the k-th confirmation arrived
    sorted_idx = sorted(fires.values())
    entry_i = int(sorted_idx[k - 1])
    px = float(bar_closes[entry_i])
    rh, rl = _ref_levels(px)
    return entry_i, px, rh, rl, len(fires)


def _find_entry_naive(bar_opens, bar_days):
    d1 = _day1_idx(bar_days)
    if len(d1) == 0:
        return -1, np.nan, np.nan, np.nan
    i = int(d1[0])
    px = float(bar_opens[i])
    rh, rl = _ref_levels(px)
    return i, px, rh, rl


def _find_entry_orb_held(bar_highs, bar_lows, bar_closes, bar_days, sig):
    """The reference: ORB_15_held_anyday_static (close > t1_orh, held EOD)."""
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


# =============================================================================
# SIMULATE ONE  (signal x variant)
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
    # bar_volumes may not be in the parquet (the v2 extractor didn't store it).
    bar_volumes = sig.get("bar_volumes")
    if bar_volumes is not None and not isinstance(bar_volumes, np.ndarray):
        bar_volumes = np.asarray(bar_volumes, dtype=float)

    engine = variant.get("engine", "feat")
    confirm_count = 0

    if engine == "alt" and variant["confirm"] == "naive_open":
        e_idx, e_px, ref_hi, ref_lo = _find_entry_naive(bar_opens, bar_days)
    elif engine == "orb":
        e_idx, e_px, ref_hi, ref_lo = _find_entry_orb_held(
            bar_highs, bar_lows, bar_closes, bar_days, sig)
    elif engine == "feat":
        e_idx, e_px, ref_hi, ref_lo = _find_entry_feat(
            bar_opens, bar_highs, bar_lows, bar_closes, bar_volumes,
            bar_days, variant, sig)
    elif engine == "conf":
        e_idx, e_px, ref_hi, ref_lo, confirm_count = _find_entry_confluence(
            bar_opens, bar_highs, bar_lows, bar_closes, bar_volumes,
            bar_days, variant, sig)
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

    for var_id, variant in VARIANTS.items():
        triggered = 0
        for i, sig in enumerate(paths_df.itertuples(index=False)):
            if i and i % 5000 == 0:
                print(f"   {var_id}: {i}/{n}  triggered={triggered}")
            sd = sig._asdict()
            res = simulate_signal(pd.Series(sd), variant)
            base = {
                "variant": var_id,
                "engine": variant.get("engine", "feat"),
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
        print(f"   {var_id}: trigger rate {triggered}/{n} = {rate:.1f}%")
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
    for keys, sub in per_trade.groupby(grp_cols):
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


# =============================================================================
# HYBRID POLICIES
# =============================================================================

HYBRID_POLICIES = {
    # the previous-Claude policy as a baseline
    "HYBRID_ORBHELD_THEN_NAIVE": {
        "[0.65,0.70)": "ORB_15_held_anyday_static",
        "[0.70,0.75)": "ORB_15_held_anyday_static",
        "[0.75,0.85)": "NAIVE_T1_OPEN",
        "[0.85,1.01)": "NAIVE_T1_OPEN",
    },
    # feature-aligned, prudent: confluence below 0.75 (where filtering matters most)
    "HYBRID_CONF2_THEN_NAIVE": {
        "[0.65,0.70)": "CONFLUENCE_2of4",
        "[0.70,0.75)": "CONFLUENCE_2of4",
        "[0.75,0.85)": "NAIVE_T1_OPEN",
        "[0.85,1.01)": "NAIVE_T1_OPEN",
    },
    # more aggressive filtering at the lowest bucket
    "HYBRID_CONF3_LOW_NAIVE_HIGH": {
        "[0.65,0.70)": "CONFLUENCE_3of4",
        "[0.70,0.75)": "CONFLUENCE_2of4",
        "[0.75,0.85)": "NAIVE_T1_OPEN",
        "[0.85,1.01)": "NAIVE_T1_OPEN",
    },
    # 52-week-high specialist: use DIST52WH_BREAK as a high-confidence gate
    "HYBRID_DIST52WH_BIAS": {
        "[0.65,0.70)": "DIST20H_BREAK",          # weaker breakout for lower bucket
        "[0.70,0.75)": "DIST52WH_BREAK",         # full breakout for higher bucket
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
    ap.add_argument("--paths-file", type=str, default=str(DEFAULT_PATHS),
                    help="extracted_paths_v2.parquet from ORB execution.py")
    ap.add_argument("--panel-file", type=str, default=str(DEFAULT_PANEL),
                    help="panel_cache.parquet (for prev-day rolling levels)")
    ap.add_argument("--out-dir",    type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--prob-min", type=float, default=SIGNAL_PROB_MIN)
    args = ap.parse_args()

    paths_path = Path(args.paths_file)
    panel_path = Path(args.panel_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not paths_path.exists():
        raise SystemExit(f"FATAL: paths file not found: {paths_path}")
    if not panel_path.exists():
        raise SystemExit(f"FATAL: panel file not found: {panel_path}")

    print(f"[main] reading paths: {paths_path}")
    paths = pd.read_parquet(paths_path)
    paths = paths[paths["probability"] >= args.prob_min].reset_index(drop=True)
    print(f"[main] {len(paths):,} signals at prob >= {args.prob_min}")

    # join the daily panel features
    paths = enrich_paths_with_panel_features(paths, panel_path)

    per_trade = run_all(paths)
    pt_path = out_dir / "per_trade_v3.parquet"
    per_trade.to_parquet(pt_path, index=False)
    print(f"[out] {pt_path}")

    summary = variant_summary(per_trade)
    summary.to_csv(out_dir / "variants_comparison_v3.csv", index=False)
    print(f"[out] {out_dir / 'variants_comparison_v3.csv'}")

    hybrid_pt = synthesize_hybrid_policies(per_trade)
    if not hybrid_pt.empty:
        hyb_summary = variant_summary(hybrid_pt)
        hyb_summary.to_csv(out_dir / "hybrid_policies_v3.csv", index=False)
        print(f"[out] {out_dir / 'hybrid_policies_v3.csv'}")
    else:
        hyb_summary = pd.DataFrame()

    try:
        with pd.ExcelWriter(out_dir / "variants_summary_v3.xlsx", engine="openpyxl") as xw:
            summary.to_excel(xw, sheet_name="variants", index=False)
            if not hyb_summary.empty:
                hyb_summary.to_excel(xw, sheet_name="hybrid", index=False)
    except Exception as e:
        print(f"[warn] xlsx write skipped: {e}")

    show = ["variant", "regime", "prob_bucket", "n_taken", "trigger_pct",
            "mean_net_pct", "win_pct", "port_ann_sharpe", "port_max_dd_pct"]
    print("\n=== variants_comparison_v3 (sorted by Sharpe per bucket) ===")
    s = summary.sort_values(["regime", "prob_bucket", "port_ann_sharpe"],
                             ascending=[True, True, False])
    print(s[show].to_string(index=False, float_format=lambda x: f"{x:7.2f}" if pd.notna(x) else "    nan"))

    if not hyb_summary.empty:
        print("\n=== hybrid_policies_v3 ===")
        print(hyb_summary[show].to_string(index=False,
              float_format=lambda x: f"{x:7.2f}" if pd.notna(x) else "    nan"))


if __name__ == "__main__":
    main()
