"""
global_cache.py
===============
US macro cache: SPY (S&P 500 proxy) + ^VIX (CBOE VIX). Mirror of Indian
`Global_cache.py` (NIFTY 50 + INDIA VIX).

The Indian original cached two symbols and emitted a panel of macro
features used by `New_model.py`. Here we do the same with US equivalents.

The output schema matches the Indian original so that the model layer
(coming in the next PR) joins on the same columns. Symbol prefixes change:

    Indian -> US (in panel features)
    M_nifty_dist_sma50    -> M_spy_dist_sma50
    M_nifty_dist_sma200   -> M_spy_dist_sma200
    M_nifty_ret_5d        -> M_spy_ret_5d
    M_vix                 -> M_vix          (same name; CBOE VIX, not INDIA VIX)
    M_vix_level_z60       -> M_vix_level_z60
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from daily_cache import (
    Config, AlpacaProvider, FileLock, with_retry,
    atomic_write_bytes, write_json_atomic, to_parquet, read_parquet, read_json,
    SCHEMA_VERSION, OK_VERSION_KEY, ok_meta_base, ET, now_et, today_et,
    DEFAULT_CACHE_DIR,
)

# alpaca-py for VIX (note: VIX is an index; Alpaca historical data is for
# stocks/ETFs only). For VIX we use the VXX or VIXY ETF as a proxy if direct
# index data is unavailable. The user can swap in a different source.
try:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment
    _ALPACA_OK = True
except Exception:
    _ALPACA_OK = False


# --------------------------------------------------------------------------
# Symbol choices
# --------------------------------------------------------------------------

SPY_SYMBOL = "SPY"        # S&P 500 ETF
VIX_PROXY_SYMBOL = "VXX"  # Volatility ETN; tracks VIX futures (not exact)
                          # If you have ^VIX from another data source,
                          # set US_VIX_SOURCE=path/to/vix.csv (date,close).

DEFAULT_GLOBAL_PATH = Path(os.environ.get("US_GLOBAL_PATH",
                            str(DEFAULT_CACHE_DIR / "macro_cache.parquet"))).expanduser()
DEFAULT_GLOBAL_OK = DEFAULT_GLOBAL_PATH.with_suffix(".ok.json")


# --------------------------------------------------------------------------
# VIX loaders (Alpaca VXX proxy or external CSV)
# --------------------------------------------------------------------------

def load_vix_from_csv(path: Path) -> pd.DataFrame:
    """External VIX CSV with columns: date, close (or value)."""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"{path}: expected 'date' column")
    val_col = "close" if "close" in df.columns else \
              "value" if "value" in df.columns else \
              "vix" if "vix" in df.columns else None
    if val_col is None:
        raise ValueError(f"{path}: expected one of close/value/vix columns")
    df["timestamp"] = pd.to_datetime(df["date"]).dt.tz_localize(ET)
    df["vix_close"] = pd.to_numeric(df[val_col], errors="coerce")
    return df[["timestamp", "vix_close"]].dropna().sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------
# Macro feature computation (mirror of Indian Global_cache.py)
# --------------------------------------------------------------------------

def compute_macro_features(spy: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """
    Combines SPY + VIX into a single daily panel:
      timestamp, M_spy_close, M_spy_sma50, M_spy_sma200,
      M_spy_dist_sma50, M_spy_dist_sma200, M_spy_ret_5d,
      M_vix_close, M_vix_sma60, M_vix_level_z60
    """
    spy = spy.copy().sort_values("timestamp").reset_index(drop=True)
    spy["M_spy_close"] = pd.to_numeric(spy["close"], errors="coerce")
    spy["M_spy_sma50"] = spy["M_spy_close"].rolling(50, min_periods=10).mean()
    spy["M_spy_sma200"] = spy["M_spy_close"].rolling(200, min_periods=50).mean()
    spy["M_spy_dist_sma50"] = (spy["M_spy_close"] / spy["M_spy_sma50"] - 1) * 100
    spy["M_spy_dist_sma200"] = (spy["M_spy_close"] / spy["M_spy_sma200"] - 1) * 100
    spy["M_spy_ret_5d"] = spy["M_spy_close"].pct_change(5) * 100
    spy = spy[["timestamp", "M_spy_close", "M_spy_sma50", "M_spy_sma200",
                "M_spy_dist_sma50", "M_spy_dist_sma200", "M_spy_ret_5d"]]

    if vix is None or vix.empty:
        spy["M_vix_close"] = np.nan
        spy["M_vix_sma60"] = np.nan
        spy["M_vix_level_z60"] = np.nan
        return spy

    vix = vix.copy().sort_values("timestamp").reset_index(drop=True)
    vix["M_vix_close"] = pd.to_numeric(vix.get("vix_close", vix.get("close")), errors="coerce")
    vix["M_vix_sma60"] = vix["M_vix_close"].rolling(60, min_periods=15).mean()
    vix["M_vix_level_z60"] = (vix["M_vix_close"] - vix["M_vix_sma60"]) / \
                                vix["M_vix_close"].rolling(60, min_periods=15).std()
    vix = vix[["timestamp", "M_vix_close", "M_vix_sma60", "M_vix_level_z60"]]

    # Date-key join (anchor to ET-normalised date)
    spy["__d"] = pd.to_datetime(spy["timestamp"]).dt.normalize()
    vix["__d"] = pd.to_datetime(vix["timestamp"]).dt.normalize()
    merged = spy.merge(vix.drop(columns=["timestamp"]), on="__d", how="left")
    merged = merged.drop(columns=["__d"]).sort_values("timestamp").reset_index(drop=True)
    return merged


# --------------------------------------------------------------------------
# Build / refresh
# --------------------------------------------------------------------------

def _fetch_etf_daily(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    provider = AlpacaProvider()
    fn = with_retry(provider.fetch_daily_bars, tries=5, backoff=0.5)
    df = fn(symbol, start, end)
    return df


def build_macro_cache(*, start: dt.date, end: dt.date,
                       out_path: Path = DEFAULT_GLOBAL_PATH,
                       force: bool = False) -> dict:
    out_ok = out_path.with_suffix(".ok.json")

    with FileLock(out_path):
        meta = read_json(out_ok)
        meta_ok = bool(meta and meta.get(OK_VERSION_KEY) == SCHEMA_VERSION)

        # SPY
        print(f"[global_cache] fetching SPY {start} -> {end}")
        spy = _fetch_etf_daily(SPY_SYMBOL, start, end)
        if spy.empty:
            return {"action": "error", "error": "SPY fetch returned empty"}

        # VIX -- prefer external CSV if provided, else VXX proxy
        vix_csv = os.environ.get("US_VIX_SOURCE")
        vix: Optional[pd.DataFrame] = None
        if vix_csv and Path(vix_csv).exists():
            print(f"[global_cache] reading VIX from {vix_csv}")
            try:
                vix = load_vix_from_csv(Path(vix_csv))
            except Exception as e:
                print(f"[global_cache] VIX CSV failed: {e}; falling back to VXX")
                vix = None
        if vix is None:
            print(f"[global_cache] fetching VXX as VIX proxy {start} -> {end}")
            vix_raw = _fetch_etf_daily(VIX_PROXY_SYMBOL, start, end)
            if not vix_raw.empty:
                vix = vix_raw[["timestamp", "close"]].copy()
                vix.columns = ["timestamp", "vix_close"]

        macro = compute_macro_features(spy, vix if vix is not None else pd.DataFrame())

        to_parquet(out_path, macro,
                    engine="pyarrow",
                    compression="snappy",
                    use_dictionary=True)

        meta = ok_meta_base()
        meta.update({
            "n_rows": int(len(macro)),
            "start": str(macro["timestamp"].min().date()),
            "end": str(macro["timestamp"].max().date()),
            "spy_symbol": SPY_SYMBOL,
            "vix_source": "csv" if (vix_csv and Path(vix_csv).exists()) else
                          ("vxx_proxy" if vix is not None else "missing"),
        })
        write_json_atomic(out_ok, meta)
        return {"action": "ok", **meta}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--out", type=str, default=str(DEFAULT_GLOBAL_PATH))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    end = pd.Timestamp(args.end).date() if args.end else today_et()
    start = pd.Timestamp(args.start).date() if args.start else \
            (end - dt.timedelta(days=int(args.years * 365.25)))

    res = build_macro_cache(start=start, end=end,
                              out_path=Path(args.out).expanduser(),
                              force=args.refresh)
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
