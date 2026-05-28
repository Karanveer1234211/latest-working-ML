"""
intraday_cache.py
=================
US-equities 5-min OHLCV cache. Mirror of Indian `latest intraday cache.py`.

Same on-disk layout (per-symbol parquet, .ok.json sidecar, atomic write,
FileLock) and same backfill semantics. Provider swapped to Alpaca and
session anchored to US regular hours (09:30-16:00 ET).

USAGE
-----
    python intraday_cache.py --years 2
    python intraday_cache.py --symbols-file syms.txt --refresh
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from universe import sanitize_symbol, filename_safe, load_universe
from daily_cache import (
    Config, FileLock, RateLimiter, with_retry,
    atomic_write_bytes, write_json_atomic, to_parquet, read_parquet, read_json,
    SCHEMA_VERSION, OK_VERSION_KEY, ok_meta_base, ET, now_et, today_et,
    DEFAULT_CACHE_DIR,
)

# alpaca-py
try:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment
    _ALPACA_OK = True
except Exception as _e:
    _ALPACA_OK = False
    _ALPACA_IMPORT_ERR = str(_e)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def _default_intraday_root() -> Path:
    env = os.environ.get("US_INTRADAY_ROOT")
    return Path(env).expanduser() if env else (DEFAULT_CACHE_DIR / "intraday_5min")


def intraday_path(root: Path, symbol: str) -> Path:
    s = sanitize_symbol(symbol) or "UNKNOWN"
    return root / f"{filename_safe(s)}.parquet"


def intraday_ok_path(root: Path, symbol: str) -> Path:
    s = sanitize_symbol(symbol) or "UNKNOWN"
    return root / f"{filename_safe(s)}.ok.json"


# --------------------------------------------------------------------------
# Provider (5-min bars from Alpaca)
# --------------------------------------------------------------------------

class AlpacaIntradayProvider:
    def __init__(self):
        if not _ALPACA_OK:
            raise RuntimeError(
                f"alpaca-py not installed. Run pip install alpaca-py. "
                f"({_ALPACA_IMPORT_ERR})"
            )
        from alpaca_auth import get_data_client, resolved_data_feed, load_creds
        creds = load_creds()
        self._client = get_data_client(creds=creds)
        self._feed = resolved_data_feed(creds)

    def fetch_5min_bars(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        if start > end:
            return pd.DataFrame()
        end_q = end + dt.timedelta(days=2)
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=dt.datetime.combine(start, dt.time()).replace(tzinfo=dt.timezone.utc),
            end=dt.datetime.combine(end_q, dt.time()).replace(tzinfo=dt.timezone.utc),
            feed=self._feed,
            adjustment=Adjustment.SPLIT,
        )
        bars = self._client.get_stock_bars(req)
        if bars is None or bars.df is None or bars.df.empty:
            return pd.DataFrame()
        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df["timestamp"] = df["timestamp"].dt.tz_convert(ET)
        keep = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]].sort_values("timestamp").reset_index(drop=True)
        # Restrict to regular session 09:30-16:00 ET
        ts = df["timestamp"]
        mask = ((ts.dt.hour > 9) | ((ts.dt.hour == 9) & (ts.dt.minute >= 30))) & \
               ((ts.dt.hour < 16) | ((ts.dt.hour == 16) & (ts.dt.minute == 0)))
        return df.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# Build / refresh
# --------------------------------------------------------------------------

def _cached_span(root: Path, symbol: str):
    p = intraday_path(root, symbol)
    if not p.exists():
        return None, None
    try:
        df = read_parquet(p, columns=["timestamp"])
    except Exception:
        return None, None
    if df.empty:
        return None, None
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(ET)
    return ts.min().date(), ts.max().date()


def _meta_current(root: Path, symbol: str) -> bool:
    meta = read_json(intraday_ok_path(root, symbol))
    return bool(meta and meta.get(OK_VERSION_KEY) == SCHEMA_VERSION)


def build_intraday(symbol: str, *, root: Path, start: dt.date, end: dt.date,
                    provider: AlpacaIntradayProvider, force: bool = False) -> dict:
    sym = sanitize_symbol(symbol)
    if not sym:
        return {"symbol": symbol, "action": "skip", "error": "invalid symbol"}

    out_path = intraday_path(root, sym)
    out_ok = intraday_ok_path(root, sym)

    with FileLock(out_path):
        cached_start, cached_end = _cached_span(root, sym)
        meta_ok = _meta_current(root, sym)

        if force or not meta_ok or cached_start is None:
            fetch_start, fetch_end = start, end
        else:
            fetch_start = cached_end + dt.timedelta(days=1)
            fetch_end = end
            if fetch_start > fetch_end:
                return {"symbol": sym, "action": "current", "n_rows": 0,
                        "start": cached_start, "end": cached_end}

        retry_fetch = with_retry(provider.fetch_5min_bars, tries=5, backoff=0.5)
        try:
            new_df = retry_fetch(sym, fetch_start, fetch_end)
        except Exception as e:
            return {"symbol": sym, "action": "error", "error": str(e)}

        if new_df.empty and cached_start is None:
            return {"symbol": sym, "action": "empty", "n_rows": 0}

        if cached_start is not None and not force:
            existing = read_parquet(out_path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"])
            if existing["timestamp"].dt.tz is None:
                existing["timestamp"] = existing["timestamp"].dt.tz_localize(ET)
            df = pd.concat([existing, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        else:
            df = new_df.sort_values("timestamp").reset_index(drop=True)

        if df.empty:
            return {"symbol": sym, "action": "empty", "n_rows": 0}

        to_parquet(out_path, df,
                    engine="pyarrow",
                    compression=os.environ.get("PARQUET_COMPRESSION", "snappy"),
                    use_dictionary=True)
        meta = ok_meta_base()
        meta.update({
            "symbol": sym,
            "n_rows": int(len(df)),
            "start": str(df["timestamp"].min().date()),
            "end": str(df["timestamp"].max().date()),
            "bar_resolution": "5min",
        })
        write_json_atomic(out_ok, meta)
        return {"symbol": sym, "action": "ok", "n_rows": int(len(df)),
                "start": meta["start"], "end": meta["end"]}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--symbols-file", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--rate-per-sec", type=float, default=30.0)
    ap.add_argument("--out-dir", type=str, default=str(_default_intraday_root()))
    args = ap.parse_args()

    root = Path(args.out_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp(args.end).date() if args.end else today_et()
    start = pd.Timestamp(args.start).date() if args.start else \
            (end - dt.timedelta(days=int(args.years * 365.25)))
    print(f"[intraday_cache] range: {start} -> {end}")

    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            syms = [sanitize_symbol(x) for x in f]
        syms = [s for s in syms if s]
    else:
        syms = load_universe()
    if args.limit > 0:
        syms = syms[: args.limit]
    print(f"[intraday_cache] universe: {len(syms)} symbols")

    provider = AlpacaIntradayProvider()
    rl = RateLimiter(args.rate_per_sec)
    results: List[dict] = []

    def _job(sym: str):
        rl.acquire()
        return build_intraday(sym, root=root, start=start, end=end,
                               provider=provider, force=args.refresh)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, s): s for s in syms}
        done = 0
        n = len(futs)
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 50 == 0 or done == n:
                ok = sum(1 for r in results if r.get("action") == "ok")
                err = sum(1 for r in results if r.get("action") == "error")
                cur = sum(1 for r in results if r.get("action") == "current")
                print(f"  [{done}/{n}]  ok={ok}  current={cur}  err={err}")

    err = sum(1 for r in results if r.get("action") == "error")
    if err:
        print("\nFirst 10 errors:")
        for r in [x for x in results if x.get("action") == "error"][:10]:
            print(f"  {r['symbol']}: {r.get('error')}")


if __name__ == "__main__":
    main()
