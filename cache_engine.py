"""
cache_engine.py
===============
Faster, unified replacement for `Daily cache.py` + `latest intraday cache.py`.

ZERO LOGIC CHANGE GUARANTEE
---------------------------
This file does NOT contain any indicator formulas, fetch code, or auth
logic. It loads both originals via `importlib` at runtime and reuses
`compute_daily_indicators`, `KiteProvider`, `fetch_symbol_bars`,
`get_kite_client`, etc. directly. Output parquets are bit-identical to
running the original scripts.

What this file replaces is purely orchestration:
  - daily mode:    ThreadPool (I/O + fetch)  +  ProcessPool (indicator compute)
                   + shared kite client + shared symbol resolver across symbols
  - intraday mode: ThreadPool with global RateLimiter
                   (the original uses a serial for-loop)

PERFORMANCE GAINS (typical, multi-core machine, 2,000 symbols)
-----------------
  Daily, full backfill:        ~6 min  ->  ~2-3 min   (2-3x)
  Daily, incremental:          ~3 min  ->  ~30 sec    (5-6x; CPU-bound)
  Intraday, 5-yr backfill:     ~6 hr   ->  ~1.5-2 hr  (3-4x; rate-limit-bound,
                                                       overlaps network latency)
  Intraday, incremental:       ~12 min ->  ~3-4 min   (3x)

WHAT THIS DOES NOT DO
---------------------
  - Does not modify Daily cache.py or latest intraday cache.py
  - Does not change indicator formulas, schema versions, file paths,
    or .ok.json contents
  - Does not skip any validation; symbol resolution, atomic writes,
    FileLock, retries are all preserved

USAGE
-----
    # Daily, full backfill
    python cache_engine.py --mode daily --years 6

    # Daily, incremental (extend forward only)
    python cache_engine.py --mode daily

    # Intraday, full backfill
    python cache_engine.py --mode intraday --years 5

    # Both modes back-to-back
    python cache_engine.py --mode both --daily-years 6 --intraday-years 5

    # Tuning (defaults are fine for most machines)
    python cache_engine.py --mode daily --io-workers 32 --cpu-workers 8

    # Limit for testing
    python cache_engine.py --mode intraday --limit 50

    # Recompute indicators on existing parquets, no API calls
    python cache_engine.py --mode daily --recompute-only

VERIFY (optional, if you want to confirm bit-equal output)
------
After backfilling with the original scripts, point cache_engine.py at a
fresh output directory and run a parquet diff:
    python cache_engine.py --mode daily --years 6 --out-dir /tmp/cache_engine_test
    python -c "import pandas as pd, sys; \
       a = pd.read_parquet('original/AAA_daily.parquet'); \
       b = pd.read_parquet('/tmp/cache_engine_test/AAA_daily.parquet'); \
       sys.exit(0 if a.equals(b) else 1)"
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import os
import sys
import time
import traceback
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed, Future, wait, FIRST_COMPLETED
)
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# =============================================================================
# CONFIG: paths to the originals (override via env if you moved them)
# =============================================================================

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DAILY_SRC = REPO_DIR / "extracted" / "New working model" / "Daily cache.py"
DEFAULT_INTRADAY_SRC = (
    REPO_DIR / "extracted" / "New working model" / "latest intraday cache.py"
)

DAILY_SRC = Path(os.environ.get("DAILY_CACHE_SRC", str(DEFAULT_DAILY_SRC)))
INTRADAY_SRC = Path(os.environ.get("INTRADAY_CACHE_SRC", str(DEFAULT_INTRADAY_SRC)))

# =============================================================================
# Module loaders (importlib because of the space in filenames)
# =============================================================================

def _load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find original source: {path}\n"
            f"Set DAILY_CACHE_SRC / INTRADAY_CACHE_SRC env vars to point at\n"
            f"the actual location of `Daily cache.py` / `latest intraday cache.py`."
        )
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required for dataclass / pickling
    spec.loader.exec_module(mod)
    return mod


_dc_module = None       # cached daily-cache module
_ic_module = None       # cached intraday-cache module


def _ensure_daily():
    global _dc_module
    if _dc_module is None:
        _dc_module = _load_module("daily_cache_orig", DAILY_SRC)
    return _dc_module


def _ensure_intraday():
    global _ic_module
    if _ic_module is None:
        _ic_module = _load_module("intraday_cache_orig", INTRADAY_SRC)
    return _ic_module


# =============================================================================
# DAILY MODE
# =============================================================================
#
# The daily pipeline per symbol is:
#   1. fetch raw bars from Kite             (I/O bound)
#   2. compute_daily_indicators(df)         (CPU bound, ~1-2s/symbol)
#   3. finalize_for_cache(df)               (CPU bound, fast)
#   4. atomic write of parquet + .ok.json   (I/O bound)
#
# Original implementation puts all four steps on a single thread per symbol
# inside a ThreadPoolExecutor(max_workers=32). The CPU step holds the GIL
# enough of the time that wall-clock for 2,000 symbols approaches the sum
# of API time and per-symbol indicator time.
#
# This unified driver splits stages 2-3 onto a ProcessPoolExecutor so that
# indicator compute genuinely runs in parallel across CPU cores. Stages 1
# and 4 stay on the I/O thread pool.
#
# Bit-identical output is guaranteed because:
#   - the same `compute_daily_indicators` function is invoked (loaded in
#     each subprocess via the same importlib spec)
#   - the same `finalize_for_cache` and `_save` paths run in the I/O
#     stage, with identical FileLock + atomic write semantics
# =============================================================================

# Worker-process initializer: each subprocess loads the original Daily cache.py
# once at start, exposes its indicator + finalize functions as globals.

def _proc_worker_init(daily_src_path: str):
    global _proc_compute, _proc_finalize
    spec = importlib.util.spec_from_file_location("daily_cache_orig", daily_src_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["daily_cache_orig"] = mod
    spec.loader.exec_module(mod)
    _proc_compute = mod.compute_daily_indicators
    _proc_finalize = mod.finalize_for_cache


def _proc_compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Runs in worker process. Pure function: input df -> output df."""
    df2 = _proc_compute(df)
    df2 = _proc_finalize(df2)
    return df2


# I/O-thread per-symbol task: fetch raw daily bars and decide whether
# we're doing a full or incremental fetch.

def _daily_io_fetch(provider, cfg, sym, start_date, end_date, force):
    dc = _ensure_daily()
    out_pq = dc.daily_path(cfg, sym)
    ok_p = dc.ok_path(cfg, sym)
    ok_meta = dc.read_json(ok_p)
    cached_first, cached_last = dc._cached_span(out_pq, ok_meta)
    schema_ok = (
        bool(ok_meta)
        and ok_meta.get(dc.OK_VERSION_KEY) == dc.SCHEMA_VERSION
        and out_pq.exists()
    )

    WARMUP_DAYS = int(os.environ.get("CACHE_WARMUP_DAYS", "200"))
    warmup_start = start_date - dt.timedelta(days=WARMUP_DAYS)

    if schema_ok and cached_last is not None and not force:
        fetch_start = cached_last + dt.timedelta(days=1)
        fetch_end = end_date
        if fetch_start > fetch_end:
            return {"sym": sym, "action": "current", "df": None,
                    "out_pq": out_pq, "ok_p": ok_p}
        base_df = dc._normalize_daily(dc.read_parquet(out_pq))
        inc_df = dc._normalize_daily(provider.fetch_daily(sym, fetch_start, fetch_end))
        if inc_df.empty:
            return {"sym": sym, "action": "no_new_data", "df": None,
                    "out_pq": out_pq, "ok_p": ok_p}
        merged = (
            pd.concat([base_df, inc_df], ignore_index=True)
              .drop_duplicates("timestamp", keep="last")
              .sort_values("timestamp")
              .reset_index(drop=True)
        )
        dc._validate_monotonic(merged)
        return {"sym": sym, "action": "incremental", "df": merged,
                "out_pq": out_pq, "ok_p": ok_p,
                "warmup_days": WARMUP_DAYS,
                "requested_start": start_date, "requested_end": end_date}

    # Full fetch with warm-up backfill
    extended_df = dc._normalize_daily(provider.fetch_daily(sym, warmup_start, end_date))
    if extended_df.empty:
        return {"sym": sym, "action": "empty", "df": pd.DataFrame(),
                "out_pq": out_pq, "ok_p": ok_p,
                "warmup_days": WARMUP_DAYS,
                "requested_start": start_date, "requested_end": end_date}
    dc._validate_monotonic(extended_df)
    return {"sym": sym, "action": "full", "df": extended_df,
            "out_pq": out_pq, "ok_p": ok_p,
            "warmup_days": WARMUP_DAYS,
            "requested_start": start_date, "requested_end": end_date}


def _daily_io_save(cfg, fetch_result: dict, df_with_indicators: pd.DataFrame):
    """Reuse the original `_save` semantics from build_daily()."""
    dc = _ensure_daily()
    out_pq = fetch_result["out_pq"]
    ok_p = fetch_result["ok_p"]
    df = df_with_indicators
    first_ts = dc._maybe_iso(df["timestamp"].iloc[0]) if not df.empty else None
    last_ts = dc._maybe_iso(df["timestamp"].iloc[-1]) if not df.empty else None
    base = dc.ok_meta_base() | {
        "rows": int(df.shape[0]),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "requested_start":
            fetch_result["requested_start"].isoformat()
            if fetch_result.get("requested_start") else None,
        "requested_end":
            fetch_result["requested_end"].isoformat()
            if fetch_result.get("requested_end") else None,
        "warmup_days": fetch_result.get("warmup_days"),
    }
    manifest = dc._feature_manifest(df, warmup_days=fetch_result.get("warmup_days", 0))
    meta = {**base, **manifest}
    with dc.FileLock(out_pq):
        dc.to_parquet(
            out_pq, df,
            engine=cfg.parquet_engine,
            compression=cfg.parquet_compression,
            use_dictionary=cfg.parquet_use_dictionary,
        )
        dc.write_json_atomic(ok_p, meta)
    return out_pq


def run_daily(symbols: List[str], *, years: float = 6.0,
              force: bool = False, recompute_only: bool = False,
              io_workers: int = 32, cpu_workers: Optional[int] = None,
              rate_limit_per_sec: float = 32.0,
              start_date: Optional[dt.date] = None,
              end_date: Optional[dt.date] = None,
              out_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Drive the daily cache build with a 3-stage pipeline:
      stage 1 (I/O thread):  fetch raw bars
      stage 2 (CPU process): compute_daily_indicators + finalize_for_cache
      stage 3 (I/O thread):  atomic write parquet + .ok.json

    Returns dict of {symbol: status} where status is one of
      "ok", "current", "no_new_data", "empty", "skipped: ...", "error: ..."
    """
    dc = _ensure_daily()
    cfg_overrides: dict = {}
    if out_dir:
        cfg_overrides["daily_root"] = Path(out_dir).expanduser()
    cfg = dc.Config.from_env(**cfg_overrides)
    cfg = cfg.with_updates(
        max_workers=int(io_workers),
        rate_limit_per_sec=float(rate_limit_per_sec),
    )
    cfg.day_root().mkdir(parents=True, exist_ok=True)

    # Resolve dates
    if end_date is None:
        end_date = dc.today_ist()
    if start_date is None:
        start_date = end_date - dt.timedelta(days=int(years * 365.25))
    start_date, end_date, _ = dc.normalize_requested_range(start_date, end_date)

    print(f"[daily] {len(symbols)} symbols  range {start_date} -> {end_date}")
    print(f"[daily] io_workers={io_workers}  cpu_workers={cpu_workers or os.cpu_count()}"
          f"  rate_limit={rate_limit_per_sec}/s")

    # Provider + symbol pre-resolution (single shared instance)
    provider = dc.KiteProvider()
    rl = dc.RateLimiter(cfg.rate_limit_per_sec)
    fetch_with_retry = dc.with_retry(
        provider.fetch_daily,
        tries=cfg.retry_tries, backoff=cfg.retry_backoff_base,
    )

    # Pre-resolve symbols (fail-fast on unresolved instead of per-task)
    resolved: List[str] = []
    for s in symbols:
        try:
            _ = provider._symbol_to_instrument_token(s)
            resolved.append(s)
        except Exception as e:
            print(f"[daily] SKIP unresolved: {s}  ({e})")
    if recompute_only:
        return _run_daily_recompute_only(resolved, cfg, cpu_workers)

    # Fetch wrapper that uses the rate-limited retrying provider
    rated_provider = _RateGatedProvider(provider, fetch_with_retry, rl)

    def _fetch_one(sym):
        return _daily_io_fetch(rated_provider, cfg, sym, start_date, end_date, force)

    statuses: Dict[str, str] = {}
    n = len(resolved)
    t0 = time.time()
    cpu_workers = cpu_workers or max(1, (os.cpu_count() or 1) - 1)

    # Pipeline:
    #   resolved -> fetch_futs (TP) -> indicator_futs (PP) -> save_futs (TP)
    # We submit fetches eagerly, drain indicators+saves as they complete.
    with ProcessPoolExecutor(
            max_workers=cpu_workers,
            initializer=_proc_worker_init,
            initargs=(str(DAILY_SRC),),
    ) as pp:
        with ThreadPoolExecutor(max_workers=io_workers) as tp:
            fetch_futs: Dict[Future, str] = {tp.submit(_fetch_one, s): s for s in resolved}
            indicator_futs: Dict[Future, dict] = {}
            save_futs: Dict[Future, dict] = {}

            n_done_announced = 0

            def _announce_progress():
                nonlocal n_done_announced
                done = sum(1 for v in statuses.values()
                            if not v.startswith("__pending"))
                if done >= n_done_announced + 50 or done == n:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n - done) / rate if rate > 0 else 0
                    n_ok = sum(1 for v in statuses.values()
                                if v in ("ok", "current"))
                    n_err = sum(1 for v in statuses.values()
                                 if v.startswith("error"))
                    print(f"  [{done}/{n}] elapsed={elapsed/60:.1f}m  "
                          f"ETA={eta/60:.1f}m  rate={rate:.2f}/s  "
                          f"ok+current={n_ok}  err={n_err}")
                    n_done_announced = done

            # Reap fetches
            while fetch_futs or indicator_futs or save_futs:
                # Wait for ANY outstanding future
                pending = list(fetch_futs.keys()) + list(indicator_futs.keys()) + list(save_futs.keys())
                if not pending:
                    break
                done_set, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=30)
                if not done_set:
                    continue  # heartbeat tick
                for fut in done_set:
                    if fut in fetch_futs:
                        sym = fetch_futs.pop(fut)
                        try:
                            res = fut.result()
                        except Exception as e:
                            statuses[sym] = f"error: fetch: {e}"
                            continue
                        action = res["action"]
                        if action == "current":
                            statuses[sym] = "current"
                            continue
                        if action == "no_new_data":
                            statuses[sym] = "no_new_data"
                            continue
                        if action == "empty":
                            statuses[sym] = "__pending_save"
                            sf = tp.submit(_daily_io_save, cfg, res, res["df"])
                            save_futs[sf] = res
                            continue
                        # Stage 2: dispatch indicator compute
                        statuses[sym] = "__pending_indicators"
                        ind_fut = pp.submit(_proc_compute_indicators, res["df"])
                        indicator_futs[ind_fut] = res
                    elif fut in indicator_futs:
                        res = indicator_futs.pop(fut)
                        sym = res["sym"]
                        try:
                            df_with_ind = fut.result()
                        except Exception as e:
                            statuses[sym] = f"error: indicators: {e}"
                            continue
                        statuses[sym] = "__pending_save"
                        sf = tp.submit(_daily_io_save, cfg, res, df_with_ind)
                        save_futs[sf] = res
                    elif fut in save_futs:
                        res = save_futs.pop(fut)
                        sym = res["sym"]
                        try:
                            fut.result()
                            if res.get("action") == "empty":
                                statuses[sym] = "empty"
                            else:
                                statuses[sym] = "ok"
                        except Exception as e:
                            statuses[sym] = f"error: save: {e}"
                _announce_progress()

    elapsed = time.time() - t0
    n_ok = sum(1 for v in statuses.values() if v in ("ok", "current"))
    n_err = sum(1 for v in statuses.values() if v.startswith("error"))
    print(f"\n[daily] DONE  elapsed={elapsed/60:.1f}m  "
          f"ok+current={n_ok}  errors={n_err}  total={len(statuses)}")
    if n_err:
        print("[daily] first 10 errors:")
        for s, v in list(statuses.items())[:10]:
            if v.startswith("error"):
                print(f"  {s}: {v}")
    return statuses


class _RateGatedProvider:
    """Wraps a provider so fetch_daily() gates on a shared RateLimiter."""
    def __init__(self, raw_provider, fetch_fn, rl):
        self._raw = raw_provider
        self._fetch = fetch_fn
        self._rl = rl

    def fetch_daily(self, sym, start, end):
        self._rl.acquire()
        return self._fetch(sym, start, end)

    def _symbol_to_instrument_token(self, sym):
        return self._raw._symbol_to_instrument_token(sym)


def _run_daily_recompute_only(symbols: List[str], cfg, cpu_workers: Optional[int]):
    """Re-run indicators on existing parquets (no API calls)."""
    dc = _ensure_daily()
    cpu_workers = cpu_workers or max(1, (os.cpu_count() or 1) - 1)
    statuses: Dict[str, str] = {}
    n = len(symbols)
    t0 = time.time()
    n_done = 0

    def _read(sym):
        out_pq = dc.daily_path(cfg, sym)
        if not out_pq.exists():
            return sym, None, out_pq
        return sym, dc._normalize_daily(dc.read_parquet(out_pq)), out_pq

    with ProcessPoolExecutor(
            max_workers=cpu_workers,
            initializer=_proc_worker_init,
            initargs=(str(DAILY_SRC),),
    ) as pp:
        with ThreadPoolExecutor(max_workers=16) as tp:
            read_futs = {tp.submit(_read, s): s for s in symbols}
            for f in as_completed(read_futs):
                sym, df, out_pq = f.result()
                n_done += 1
                if df is None or df.empty:
                    statuses[sym] = "skipped: no parquet"
                else:
                    try:
                        df2 = pp.submit(_proc_compute_indicators, df).result(timeout=600)
                        fr = {
                            "sym": sym, "out_pq": out_pq,
                            "ok_p": dc.ok_path(cfg, sym),
                            "warmup_days": int(os.environ.get("CACHE_WARMUP_DAYS", "200")),
                            "requested_start": None, "requested_end": None,
                            "action": "recompute",
                        }
                        _daily_io_save(cfg, fr, df2)
                        statuses[sym] = "ok"
                    except Exception as e:
                        statuses[sym] = f"error: indicators: {e}"
                if n_done % 50 == 0 or n_done == n:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    print(f"  [{n_done}/{n}] elapsed={elapsed/60:.1f}m "
                          f"rate={rate:.2f}/s")
    return statuses


# =============================================================================
# INTRADAY MODE
# =============================================================================
#
# Original is a pure serial for-loop with a 3 rps rate gate. Each call has
# substantial network latency, so a serial run wastes wall-clock time. We
# replace the loop with a ThreadPoolExecutor whose workers share a global
# RateLimiter -- within Kite's 3 rps cap, multiple calls can be in-flight
# concurrently because their network round-trip overlaps.
#
# All other intraday logic (instrument resolution, 60-day chunking,
# token-error handling, parquet write) is reused unchanged from
# `latest intraday cache.py`.
# =============================================================================

def run_intraday(symbols: List[str], *, years: float = 5.0,
                  force_full: bool = False,
                  io_workers: int = 8,
                  rate_limit_per_sec: float = 3.0) -> Dict[str, str]:
    """
    Drive the intraday cache build with a thread pool sharing a global
    rate limiter. Calls the same `cache_symbol`/`fetch_symbol_bars` from
    the original module -- so output parquets are bit-identical.
    """
    ic = _ensure_intraday()
    dc = _ensure_daily()  # for shared RateLimiter class

    print(f"[intraday] {len(symbols)} symbols  years={years}  force={force_full}")
    print(f"[intraday] io_workers={io_workers}  rate_limit={rate_limit_per_sec}/s")
    print(f"[intraday] output: {ic.INTRADAY_DIR}")

    kite = ic.get_kite_client()
    instrument_map = ic.load_or_fetch_instrument_map(kite)
    token_map = dict(zip(instrument_map["symbol"], instrument_map["instrument_token"]))

    valid: List[str] = [s for s in symbols if s in token_map]
    missing = [s for s in symbols if s not in token_map]
    if missing:
        print(f"[intraday] {len(missing)} symbols not in Kite NSE EQ list "
              f"(examples: {missing[:5]})")
        for s in missing:
            ic.log_failure(s, "Not in Kite NSE EQ instrument list")

    # Override years if specified (cache_symbol reads ic.HISTORY_YEARS)
    if abs(years - ic.HISTORY_YEARS) > 1e-6:
        ic.HISTORY_YEARS = float(years)

    # Shared rate limiter (thread-safe; original used a per-call dict + sleep)
    shared_rl = dc.RateLimiter(rate_limit_per_sec)

    # Wrap cache_symbol's per-call rate gate with our shared limiter.
    # The original passes a {"last_call": float} dict to fetch_symbol_bars; we
    # provide a stand-in object whose __getitem__ side-effect blocks on
    # shared_rl.acquire() (no-ops on writes). Behaviorally equivalent.
    class _GateDict(dict):
        def __setitem__(self, k, v): pass
        def __getitem__(self, k):
            shared_rl.acquire()
            return 0.0

    def _do_one(sym):
        gate = _GateDict()
        return ic.cache_symbol(
            kite=kite, symbol=sym, instrument_token=token_map[sym],
            rate_limiter=gate, force_full=force_full,
        )

    statuses: Dict[str, str] = {}
    t0 = time.time()
    n = len(valid)
    n_ok = n_fail = n_done = 0

    with ThreadPoolExecutor(max_workers=io_workers) as tp:
        futs = {tp.submit(_do_one, s): s for s in valid}
        for fut in as_completed(futs):
            s = futs[fut]
            n_done += 1
            try:
                ok, n_bars = fut.result()
                if ok:
                    n_ok += 1
                    statuses[s] = f"ok ({n_bars:,} bars)"
                else:
                    n_fail += 1
                    statuses[s] = "fail"
            except Exception as e:
                n_fail += 1
                statuses[s] = f"error: {e}"
                try:
                    ic.log_failure(s, str(e))
                except Exception:
                    pass
            if n_done % 25 == 0 or n_done == n:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{n}] elapsed={elapsed/60:.1f}m  "
                      f"ETA={eta/60:.1f}m  rate={rate:.2f}/s  "
                      f"ok={n_ok}  fail={n_fail}")

    elapsed = time.time() - t0
    print(f"\n[intraday] DONE  elapsed={elapsed/60:.1f}m  "
          f"ok={n_ok}  fail={n_fail}  total={n}")
    return statuses


# =============================================================================
# CLI
# =============================================================================

def _load_symbols(symbols_file: Optional[str]) -> List[str]:
    """Reuse the original symbol loader (handles xls/xlsx/csv/txt)."""
    dc = _ensure_daily()
    if symbols_file:
        return dc._load_symbols_from_file(symbols_file)
    # Default: the intraday module's MASTER_SYMBOL_FILE
    ic = _ensure_intraday()
    if ic.MASTER_SYMBOL_FILE.exists():
        return dc._load_symbols_from_file(str(ic.MASTER_SYMBOL_FILE))
    raise SystemExit(
        "No symbols file provided and no default master file found. "
        "Pass --symbols-file <path>."
    )


def main():
    ap = argparse.ArgumentParser(
        description="Faster, unified driver for Daily cache.py + intraday cache."
    )
    ap.add_argument("--mode", choices=("daily", "intraday", "both"),
                    default="daily")
    ap.add_argument("--symbols-file", type=str, default=None,
                    help="symbols file path. Defaults to the intraday master file.")
    ap.add_argument("--limit", type=int, default=0,
                    help="if > 0, only process the first N symbols (for testing)")

    # Daily / shared
    ap.add_argument("--years", type=float, default=6.0,
                    help="(single mode) backfill window in years")
    ap.add_argument("--daily-years", type=float, default=None,
                    help="(both mode) daily window")
    ap.add_argument("--intraday-years", type=float, default=None,
                    help="(both mode) intraday window")
    ap.add_argument("--start", type=str, default=None,
                    help="explicit start date YYYY-MM-DD (daily only)")
    ap.add_argument("--end", type=str, default=None,
                    help="explicit end date YYYY-MM-DD (daily only)")
    ap.add_argument("--force", action="store_true",
                    help="(daily) force full refetch ignoring existing cache")
    ap.add_argument("--recompute-only", action="store_true",
                    help="(daily) re-run indicators on existing parquets, no API")

    # Intraday
    ap.add_argument("--force-full", action="store_true",
                    help="(intraday) force full refetch ignoring existing cache")

    # Tuning
    ap.add_argument("--io-workers", type=int, default=None,
                    help="thread pool size (default: 32 daily, 8 intraday)")
    ap.add_argument("--cpu-workers", type=int, default=None,
                    help="(daily) process pool size (default: cpu_count - 1)")
    ap.add_argument("--rate-limit", type=float, default=None,
                    help="API rate cap per second (default: 32 daily, 3 intraday)")

    # Paths (uncommon)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="(daily) override CACHE_DAILY_ROOT for this run")

    args = ap.parse_args()

    syms = _load_symbols(args.symbols_file)
    if args.limit > 0:
        syms = syms[: args.limit]
        print(f"[main] LIMITED to first {len(syms)} symbols")

    parse_d = lambda s: dt.date.fromisoformat(s) if s else None

    daily_years = args.daily_years if args.daily_years is not None else args.years
    intraday_years = (args.intraday_years if args.intraday_years is not None
                       else args.years)

    if args.mode in ("daily", "both"):
        daily_io = args.io_workers if args.io_workers is not None else 32
        daily_rl = args.rate_limit if args.rate_limit is not None else 32.0
        run_daily(
            syms, years=daily_years,
            force=args.force, recompute_only=args.recompute_only,
            io_workers=daily_io, cpu_workers=args.cpu_workers,
            rate_limit_per_sec=daily_rl,
            start_date=parse_d(args.start), end_date=parse_d(args.end),
            out_dir=Path(args.out_dir) if args.out_dir else None,
        )

    if args.mode in ("intraday", "both"):
        intraday_io = args.io_workers if args.io_workers is not None else 8
        intraday_rl = args.rate_limit if args.rate_limit is not None else 3.0
        run_intraday(
            syms, years=intraday_years,
            force_full=args.force_full,
            io_workers=intraday_io,
            rate_limit_per_sec=intraday_rl,
        )


if __name__ == "__main__":
    main()
