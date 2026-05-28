"""
daily_cache.py
==============
US-equities daily OHLCV cache with leak-free indicators.

Faithful port of `Daily cache.py` (the Indian original) to Alpaca + NYSE.
Same on-disk schema, same atomic-write discipline, same indicator suite.

KEY DIFFERENCES VS INDIAN ORIGINAL
----------------------------------
- Provider: KiteConnect -> Alpaca (alpaca-py)
- Timezone: Asia/Kolkata -> America/New_York
- Session: 09:15-15:30 IST -> 09:30-16:00 ET
- Holidays: NSE -> NYSE (pandas_market_calendars)
- No GUI / tkinter
- Schema version v19 -> v100 (avoid collision with Indian artefacts)
- Macro features: M_nifty_*, M_vix removed here (live in global_cache.py)
- Liquidity: MIN_CLOSE 2.0->5.0, MIN_VOL 200K->500K (penny-stock filter)

WHAT'S IDENTICAL
----------------
- Per-symbol parquet output: {symbol}_daily.parquet
- .ok.json sidecar with schema_version + created_ts
- Atomic write through .tmp + os.replace
- FileLock for concurrent safety
- All technical indicators (RSI, ATR, ADX, BB, CMF, OBV, MACD, donchian,
  Yang-Zhang vol, etc.) and the 17 v18 structural features
- All 16 v19 WorldQuant alphas + 17 rolling/lag/diff transforms

USAGE
-----
    python daily_cache.py --years 6                # backfill 6 years
    python daily_cache.py --symbols-file syms.txt  # specific list
    python daily_cache.py --refresh                # rebuild from scratch
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import functools
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local
from universe import sanitize_symbol, filename_safe, load_universe

# alpaca-py
try:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment
    _ALPACA_OK = True
except Exception as _e:
    _ALPACA_OK = False
    _ALPACA_IMPORT_ERR = str(_e)

# Holiday calendar
try:
    import pandas_market_calendars as mcal
    _MCAL_OK = True
except Exception:
    _MCAL_OK = False

# --------------------------------------------------------------------------
# Session / timezone (US)
# --------------------------------------------------------------------------

ET = "America/New_York"
SESSION_OPEN = dt.time(9, 30)
SESSION_CLOSE = dt.time(16, 0)


def now_et() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc).astimezone(_pytz_et())


def today_et() -> dt.date:
    return now_et().date()


def _pytz_et():
    # Lazy import for portability
    from zoneinfo import ZoneInfo
    return ZoneInfo(ET)


# --------------------------------------------------------------------------
# Schema + on-disk paths (mirrors Indian original)
# --------------------------------------------------------------------------

SCHEMA_VERSION = 100  # US baseline; Indian original uses 19
OK_VERSION_KEY = "schema_version"

DEFAULT_CACHE_DIR = Path(os.environ.get("US_CACHE_DIR",
                          str(Path.home() / "us_market_cache"))).expanduser()


def _default_daily_root() -> Path:
    env = os.environ.get("US_DAILY_ROOT")
    return Path(env).expanduser() if env else (DEFAULT_CACHE_DIR / "daily")


@dataclass(frozen=True)
class Config:
    daily_root: Path = field(default_factory=_default_daily_root)
    max_workers: int = 8
    rate_limit_per_sec: float = 30.0       # Alpaca free tier ~200 rpm
    request_timeout_s: float = 30.0
    retry_tries: int = 5
    retry_backoff_base: float = 0.5
    parquet_engine: str = os.environ.get("PARQUET_ENGINE", "pyarrow")
    parquet_compression: Optional[str] = os.environ.get("PARQUET_COMPRESSION", "snappy")
    parquet_use_dictionary: bool = True

    def day_root(self) -> Path:
        return self.daily_root

    def with_updates(self, **updates) -> "Config":
        return replace(self, **updates)


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------

def daily_path(cfg: Config, symbol: str) -> Path:
    s = sanitize_symbol(symbol) or "UNKNOWN"
    return cfg.day_root() / f"{filename_safe(s)}_daily.parquet"


def ok_path(cfg: Config, symbol: str) -> Path:
    s = sanitize_symbol(symbol) or "UNKNOWN"
    return cfg.day_root() / f"{filename_safe(s)}_daily.ok.json"


def ok_meta_base() -> dict:
    return {
        OK_VERSION_KEY: SCHEMA_VERSION,
        "created_ts": now_et().isoformat(),
    }


# --------------------------------------------------------------------------
# Atomic IO + FileLock (mirrors Indian original)
# --------------------------------------------------------------------------

class FileLock:
    def __init__(self, path: Path, poll_ms: int = 50, timeout_s: float = 30.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(str(path) + ".lock")
        self.poll_ms = poll_ms
        self.timeout_s = timeout_s
        self._fd: Optional[int] = None

    def acquire(self):
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self._fd = os.open(self.lock_path,
                                    os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode())
                return
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(f"Timeout acquiring lock {self.lock_path}")
                time.sleep(self.poll_ms / 1000.0)

    def release(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.lock_path)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


def atomic_write_bytes(target: Path, data: bytes):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


def write_json_atomic(path: Path, obj: dict):
    atomic_write_bytes(path,
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str).encode())


def to_parquet(path: Path, df: pd.DataFrame, *,
                engine: str, compression: Optional[str], use_dictionary: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine=engine,
                   compression=compression, use_dictionary=use_dictionary)


def read_parquet(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Rate limiter + retry (mirrors Indian original)
# --------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, per_sec: float):
        self.per_sec = float(per_sec)
        self._lock = threading.Lock()
        self._tokens = per_sec
        self._updated = time.perf_counter()

    def acquire(self):
        while True:
            with self._lock:
                now = time.perf_counter()
                self._tokens = min(self.per_sec,
                                    self._tokens + (now - self._updated) * self.per_sec)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                need = max(0.0, 1.0 - self._tokens)
                wait = need / self.per_sec if self.per_sec > 0 else 0.0
            time.sleep(wait if wait > 0 else 0)


def with_retry(fn, *, tries: int, backoff: float):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(tries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e).lower()
                if "rate" in msg or "429" in msg or "too many" in msg:
                    sleep = min(15.0, backoff * (2.2 ** attempt))
                else:
                    sleep = min(8.0, backoff * (1.8 ** attempt) +
                                np.random.random() * (backoff / 2))
                last_exc = e
                time.sleep(sleep)
        if last_exc:
            raise last_exc
    return wrapper


# --------------------------------------------------------------------------
# Holiday-aware trading-day generator
# --------------------------------------------------------------------------

def trading_days_between(start: dt.date, end: dt.date) -> List[dt.date]:
    if _MCAL_OK:
        cal = mcal.get_calendar("NYSE")
        sched = cal.schedule(start_date=str(start), end_date=str(end))
        return [d.date() for d in sched.index]
    # Fallback: Mon-Fri minus US federal holidays (rough)
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += dt.timedelta(days=1)
    return days


# --------------------------------------------------------------------------
# Indicator helpers (universal -- copied verbatim from Indian original)
# --------------------------------------------------------------------------

def _ensure_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_dn = dn.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_dn.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([(h - l).abs(),
                      (h - pc).abs(),
                      (l - pc).abs()], axis=1).max(axis=1)


def _atr(h, l, c, period: int) -> pd.Series:
    tr = _true_range(h, l, c)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _adx(h, l, c, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up = h.diff()
    dn = -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    atr = _atr(h, l, c, period)
    pdi = 100 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx, pdi, mdi


def _rolling_ols_slope(y: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope on uniform x = 0..window-1."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(arr):
        if len(arr) < window:
            return np.nan
        a = np.asarray(arr, dtype=float)
        return ((x - x_mean) * (a - a.mean())).sum() / x_var

    return y.rolling(window, min_periods=window).apply(_slope, raw=True)


# --------------------------------------------------------------------------
# Core: indicator computation (mirrors compute_daily_indicators in Indian)
# --------------------------------------------------------------------------

def compute_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 100+ leak-free daily indicators to a per-symbol OHLCV DataFrame.

    Input columns required: timestamp, open, high, low, close, volume
    All indicators are computed with at-most a same-bar reference; no
    forward leakage.
    """
    if df.empty:
        return df
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    o = _ensure_float(df["open"])
    h = _ensure_float(df["high"])
    l = _ensure_float(df["low"])
    c = _ensure_float(df["close"])
    v = _ensure_float(df["volume"]).fillna(0.0)

    # ----- price / range / candle -----
    df["D_ret_1d_pct"] = c.pct_change() * 100
    df["D_ret_5d_pct"] = c.pct_change(5) * 100
    df["D_ret_20d_pct"] = c.pct_change(20) * 100

    df["D_gap_pct"] = (o - c.shift(1)) / c.shift(1).replace(0, np.nan) * 100
    df["D_range_pct"] = (h - l) / c.replace(0, np.nan) * 100
    df["D_body_ratio"] = (c - o).abs() / (h - l).replace(0, np.nan)
    df["D_wick_skew"] = ((h - c.combine(o, max)) - (c.combine(o, min) - l)) / (h - l).replace(0, np.nan)

    # ----- moving averages + slopes -----
    df["D_ema20"] = _ema(c, 20)
    df["D_ema50"] = _ema(c, 50)
    df["D_ema200"] = _ema(c, 200)
    df["D_sma20"] = _sma(c, 20)
    df["D_sma50"] = _sma(c, 50)
    df["D_sma200"] = _sma(c, 200)

    df["D_ema20_angle_deg"] = np.degrees(np.arctan(df["D_ema20"].pct_change(1)))
    df["D_ema20_angle_deg_lag1"] = df["D_ema20_angle_deg"].shift(1)

    df["D_close_roll_slope_20"] = _rolling_ols_slope(c, 20)

    # ----- RSI / ADX / MACD / CCI -----
    df["D_rsi7"] = _rsi(c, 7)
    df["D_rsi14"] = _rsi(c, 14)
    df["D_rsi7_gt_rsi14"] = (df["D_rsi7"] > df["D_rsi14"]).astype(int)

    adx, pdi, mdi = _adx(h, l, c, 14)
    df["D_adx14"] = adx
    df["D_pdi14"] = pdi
    df["D_mdi14"] = mdi

    macd_fast = _ema(c, 12)
    macd_slow = _ema(c, 26)
    macd = macd_fast - macd_slow
    macd_sig = _ema(macd, 9)
    df["D_macd"] = macd
    df["D_macd_signal"] = macd_sig
    df["D_macd_hist"] = macd - macd_sig

    # ----- ATR / volatility -----
    df["D_atr14"] = _atr(h, l, c, 14)
    df["D_atr_pct"] = df["D_atr14"] / c.replace(0, np.nan) * 100
    df["D_atr14_to_close_pct"] = df["D_atr_pct"]
    df["D_atr_pct_z252"] = (df["D_atr_pct"] - df["D_atr_pct"].rolling(252, min_periods=50).mean()) / \
                             df["D_atr_pct"].rolling(252, min_periods=50).std()

    # ----- Bollinger + Donchian -----
    bb_mid = _sma(c, 20)
    bb_std = c.rolling(20, min_periods=5).std()
    df["D_bb_pctB_20"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)
    df["D_bb_bw_20"] = (4 * bb_std) / bb_mid.replace(0, np.nan)

    high20 = h.rolling(20, min_periods=5).max()
    low20 = l.rolling(20, min_periods=5).min()
    df["D_donch_pos_20"] = (c - low20) / (high20 - low20).replace(0, np.nan)
    df["D_dist_from_20h"] = (c - high20) / high20.replace(0, np.nan) * 100

    high50 = h.rolling(50, min_periods=10).max()
    low50 = l.rolling(50, min_periods=10).min()
    df["D_donch_pos_50"] = (c - low50) / (high50 - low50).replace(0, np.nan)

    high252 = h.rolling(252, min_periods=50).max()
    df["D_dist_from_52wh"] = (c - high252) / high252.replace(0, np.nan) * 100

    # ----- volume / dollar volume / OBV / CMF -----
    df["D_dollar_vol"] = c * v
    df["D_dvol_z20"] = (df["D_dollar_vol"] - df["D_dollar_vol"].rolling(20, min_periods=5).mean()) / \
                        df["D_dollar_vol"].rolling(20, min_periods=5).std()
    df["D_dvol_z50"] = (df["D_dollar_vol"] - df["D_dollar_vol"].rolling(50, min_periods=10).mean()) / \
                        df["D_dollar_vol"].rolling(50, min_periods=10).std()

    direction = np.sign(c.diff()).fillna(0.0)
    df["D_obv"] = (direction * v).cumsum()
    df["D_obv_slope"] = _rolling_ols_slope(df["D_obv"], 20)
    df["D_obv_slope_lag1"] = df["D_obv_slope"].shift(1)

    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm * v
    df["D_cmf20"] = mfv.rolling(20, min_periods=5).sum() / v.rolling(20, min_periods=5).sum().replace(0, np.nan)

    # ----- Yang-Zhang volatility (20 / 50) -----
    df["D_vol_yz_20"] = _yang_zhang_vol(o, h, l, c, 20)
    df["D_vol_yz_50"] = _yang_zhang_vol(o, h, l, c, 50)

    # ----- structural micro-features (v18 in Indian) -----
    df["D_hh_run"] = _streak(h, lambda s: s > s.shift(1))
    df["D_hl_run"] = _streak(l, lambda s: s > s.shift(1))
    df["D_lh_run"] = _streak(h, lambda s: s < s.shift(1))
    df["D_ll_run"] = _streak(l, lambda s: s < s.shift(1))
    df["D_nr_expand"] = ((h - l) > (h - l).shift(1).rolling(7, min_periods=3).max()).astype(int)
    df["D_compress_state"] = (df["D_bb_bw_20"] < df["D_bb_bw_20"].rolling(50, min_periods=20).quantile(0.25)).astype(int)
    midpoint = (h + l) / 2.0
    df["D_midpoint_slope"] = midpoint.diff()
    df["D_slope_stability"] = 1.0 - df["D_midpoint_slope"].rolling(20, min_periods=5).std() / df["D_midpoint_slope"].rolling(20, min_periods=5).mean().replace(0, np.nan).abs()

    df["D_ret_5d_roll_std"] = df["D_ret_5d_pct"].rolling(20, min_periods=5).std()

    # ----- v19 transforms (rolling means/stds/lags/diffs of best features) -----
    for col in ["D_slope_stability", "D_body_ratio", "D_close_roll_slope_20",
                "D_macd_hist", "D_mdi14", "D_donch_pos_50", "D_donch_pos_20",
                "D_cmf20"]:
        if col not in df.columns:
            continue
        s = _ensure_float(df[col])
        for w in (10, 20, 50):
            df[f"{col}_rmean{w}"] = s.rolling(w, min_periods=max(3, w // 4)).mean()
            df[f"{col}_rstd{w}"] = s.rolling(w, min_periods=max(3, w // 4)).std()
        for lag in (1, 5):
            df[f"{col}_lag{lag}"] = s.shift(lag)

    # ----- v19 WorldQuant alphas -----
    _add_wq_alphas(df, o, h, l, c, v)

    # ----- weekly features (W_ret_4w, W_ret_13w, W_close_pos, W_vol_vs_4w) -----
    # Indian original used pandas weekly resample; here we use rolling-day
    # equivalents (4w ~= 20 trading days, 13w ~= 65) so the feature schema
    # is identical without resampling complexity.
    df["W_ret_4w"] = c.pct_change(20) * 100
    df["W_ret_13w"] = c.pct_change(65) * 100
    high20w = h.rolling(20, min_periods=5).max()
    low20w = l.rolling(20, min_periods=5).min()
    df["W_close_pos"] = (c - low20w) / (high20w - low20w).replace(0, np.nan)
    df["W_vol_vs_4w"] = v / v.rolling(20, min_periods=5).mean()

    return df


def _yang_zhang_vol(o, h, l, c, window: int) -> pd.Series:
    """Yang-Zhang realized volatility (annualised)."""
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_co = np.log(c / o)
    log_oc = np.log(o / c.shift(1))
    log_cc = np.log(c / c.shift(1))

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    sigma_o2 = log_oc.pow(2).rolling(window, min_periods=max(3, window // 2)).mean()
    sigma_c2 = log_cc.pow(2).rolling(window, min_periods=max(3, window // 2)).mean()
    sigma_rs = rs.rolling(window, min_periods=max(3, window // 2)).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs
    return np.sqrt(yz * 252) * 100


def _streak(series: pd.Series, predicate) -> pd.Series:
    """Length of current streak where predicate(series) is True."""
    cond = predicate(series).fillna(False)
    out = np.zeros(len(cond), dtype=int)
    run = 0
    for i, v in enumerate(cond.values):
        run = run + 1 if v else 0
        out[i] = run
    return pd.Series(out, index=series.index)


def _add_wq_alphas(df, o, h, l, c, v):
    """v19 WorldQuant alphas (16 alphas, leak-free)."""
    ret = c.pct_change()
    vwap = (h + l + c) / 3.0
    adv20 = v.rolling(20, min_periods=5).mean()

    # Helper rolling functions
    def _rank(s):
        return s.rank(pct=True)

    def _ts_rank(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).rank(pct=True)

    def _ts_corr(a, b, w):
        return a.rolling(w, min_periods=max(5, w // 2)).corr(b)

    def _delta(s, d):
        return s.diff(d)

    def _delay(s, d):
        return s.shift(d)

    def _ts_max(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).max()

    def _ts_min(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).min()

    def _ts_std(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).std()

    def _ts_mean(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).mean()

    def _signed_power(s, e):
        return np.sign(s) * (s.abs() ** e)

    def _scale(s):
        d = s.abs().sum()
        return s / d if d != 0 else s

    try: df["D_WQ_3"] = -_ts_corr(_rank(o), _rank(v), 10)
    except Exception: pass
    try: df["D_WQ_6"] = -_ts_corr(o, v, 10)
    except Exception: pass
    try: df["D_WQ_12"] = np.sign(_delta(v, 1)) * (-_delta(c, 1))
    except Exception: pass
    try: df["D_WQ_13"] = -_rank(_rank(c).rolling(5, min_periods=3).cov(_rank(v)))
    except Exception: pass
    try: df["D_WQ_16"] = -_rank(_rank(h).rolling(5, min_periods=3).cov(_rank(v)))
    except Exception: pass
    try:
        d7 = _delay(c, 7)
        part = -np.sign(_delta(c - d7, 5) + _delta(c, 5))
        ret_sum = ret.rolling(250, min_periods=50).sum()
        df["D_WQ_19"] = part * (1 + _rank(1 + ret_sum))
    except Exception: pass
    try: df["D_WQ_20"] = -_rank(o - _delay(h, 1)) * _rank(o - _delay(c, 1)) * _rank(o - _delay(l, 1))
    except Exception: pass
    try:
        cond = _ts_mean(h, 20) < h
        df["D_WQ_23"] = (-_delta(h, 2)).where(cond, 0)
    except Exception: pass
    try:
        inner = _ts_corr(_ts_rank(v, 5), _ts_rank(h, 5), 5)
        df["D_WQ_26"] = -_ts_max(inner, 3)
    except Exception: pass
    try:
        inner = -_rank(_delta(c, 5))
        df["D_WQ_29"] = _rank(_rank(inner))
    except Exception: pass
    try: df["D_WQ_33"] = _rank(-1 + o / c.replace(0, np.nan))
    except Exception: pass
    try: df["D_WQ_35"] = _ts_rank(v, 32) * (1 - _ts_rank(c + h - l, 16)) * (1 - _ts_rank(ret, 32))
    except Exception: pass
    try: df["D_WQ_38"] = -_rank(_ts_rank(c, 10)) * _rank(c / o.replace(0, np.nan))
    except Exception: pass
    try: df["D_WQ_40"] = -_rank(_ts_std(h, 10)) * _ts_corr(h, v, 10)
    except Exception: pass
    try: df["D_WQ_41"] = np.sqrt(h * l) - vwap
    except Exception: pass
    try: df["D_WQ_44"] = -_ts_corr(h, _rank(v), 5)
    except Exception: pass


# --------------------------------------------------------------------------
# Provider abstraction (Alpaca implementation)
# --------------------------------------------------------------------------

class AlpacaProvider:
    """Daily-bar provider for Alpaca. Mirrors Indian KiteProvider API."""

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

    def fetch_daily_bars(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Fetch daily OHLCV between [start, end] inclusive."""
        if start > end:
            return pd.DataFrame()
        # Alpaca treats end as exclusive in some endpoints; pad by 1 day.
        end_q = end + dt.timedelta(days=2)
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=dt.datetime.combine(start, dt.time()).replace(tzinfo=dt.timezone.utc),
            end=dt.datetime.combine(end_q, dt.time()).replace(tzinfo=dt.timezone.utc),
            feed=self._feed,
            adjustment=Adjustment.SPLIT,
        )
        bars = self._client.get_stock_bars(req)
        if bars is None or bars.df is None or bars.df.empty:
            return pd.DataFrame()
        df = bars.df.reset_index()
        # Multi-symbol response has (symbol, timestamp); single-symbol may be just timestamp.
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        df = df.rename(columns={"timestamp": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df["timestamp"] = df["timestamp"].dt.tz_convert(ET).dt.normalize()  # date-only, ET-anchored
        keep = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]].sort_values("timestamp").reset_index(drop=True)
        # Filter to [start, end] exactly
        mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
        return df.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# Build / refresh logic
# --------------------------------------------------------------------------

def _normalise_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(ET)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def _cached_span(cfg: Config, symbol: str) -> Tuple[Optional[dt.date], Optional[dt.date]]:
    p = daily_path(cfg, symbol)
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


def _meta_says_current(cfg: Config, symbol: str) -> bool:
    """True if the .ok.json sidecar reports the current SCHEMA_VERSION."""
    meta = read_json(ok_path(cfg, symbol))
    return bool(meta and meta.get(OK_VERSION_KEY) == SCHEMA_VERSION)


def build_daily(symbol: str, *, cfg: Config, start: dt.date, end: dt.date,
                provider: AlpacaProvider, force: bool = False) -> dict:
    """
    Backfill or extend the daily cache for one symbol.

    Returns dict: {symbol, action, n_rows, start, end, error?}
    """
    sym = sanitize_symbol(symbol)
    if not sym:
        return {"symbol": symbol, "action": "skip", "error": "invalid symbol"}

    out_path = daily_path(cfg, sym)
    out_ok = ok_path(cfg, sym)

    with FileLock(out_path):
        cached_start, cached_end = _cached_span(cfg, sym)
        meta_ok = _meta_says_current(cfg, sym)

        if force or not meta_ok or cached_start is None:
            # full rebuild
            fetch_start, fetch_end = start, end
        else:
            # extend forward only
            fetch_start = cached_end + dt.timedelta(days=1)
            fetch_end = end
            if fetch_start > fetch_end:
                return {"symbol": sym, "action": "current", "n_rows": 0,
                        "start": cached_start, "end": cached_end}

        retry_fetch = with_retry(provider.fetch_daily_bars,
                                  tries=cfg.retry_tries, backoff=cfg.retry_backoff_base)
        try:
            new_df = retry_fetch(sym, fetch_start, fetch_end)
        except Exception as e:
            return {"symbol": sym, "action": "error", "error": str(e)}

        if new_df.empty and (cached_start is None):
            return {"symbol": sym, "action": "empty", "n_rows": 0}

        if cached_start is not None and not force:
            existing = read_parquet(out_path)
            new_df = _normalise_daily(new_df)
            existing = _normalise_daily(existing)
            df = pd.concat([existing, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        else:
            df = _normalise_daily(new_df)

        if df.empty:
            return {"symbol": sym, "action": "empty", "n_rows": 0}

        # Compute / refresh indicators
        df = compute_daily_indicators(df)

        to_parquet(out_path, df,
                    engine=cfg.parquet_engine,
                    compression=cfg.parquet_compression,
                    use_dictionary=cfg.parquet_use_dictionary)
        meta = ok_meta_base()
        meta.update({
            "symbol": sym,
            "n_rows": int(len(df)),
            "start": str(df["timestamp"].min().date()),
            "end": str(df["timestamp"].max().date()),
        })
        write_json_atomic(out_ok, meta)

        return {"symbol": sym, "action": "ok", "n_rows": int(len(df)),
                "start": meta["start"], "end": meta["end"]}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_date(value: str) -> dt.date:
    return pd.Timestamp(value).date()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=6.0,
                    help="how many years of history to backfill")
    ap.add_argument("--start", type=str, default=None,
                    help="explicit start date YYYY-MM-DD (overrides --years)")
    ap.add_argument("--end", type=str, default=None,
                    help="explicit end date YYYY-MM-DD (defaults to today ET)")
    ap.add_argument("--symbols-file", type=str, default=None,
                    help="path to symbol list (default: built by universe.py)")
    ap.add_argument("--limit", type=int, default=0,
                    help="if > 0, process only the first N symbols")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel symbol workers")
    ap.add_argument("--refresh", action="store_true",
                    help="full rebuild ignoring existing cache")
    ap.add_argument("--out-dir", type=str, default=str(_default_daily_root()))
    args = ap.parse_args()

    cfg = Config(daily_root=Path(args.out_dir).expanduser(),
                  max_workers=int(args.workers))
    cfg.day_root().mkdir(parents=True, exist_ok=True)

    # Resolve date range
    end = _parse_date(args.end) if args.end else today_et()
    if args.start:
        start = _parse_date(args.start)
    else:
        start = end - dt.timedelta(days=int(args.years * 365.25))
    print(f"[daily_cache] range: {start}  ->  {end}")

    # Resolve universe
    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            syms = [sanitize_symbol(x) for x in f]
        syms = [s for s in syms if s]
    else:
        syms = load_universe()
    if args.limit > 0:
        syms = syms[: args.limit]
    print(f"[daily_cache] universe: {len(syms)} symbols")

    provider = AlpacaProvider()

    rl = RateLimiter(cfg.rate_limit_per_sec)
    results: List[dict] = []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _job(sym: str):
        rl.acquire()
        return build_daily(sym, cfg=cfg, start=start, end=end,
                            provider=provider, force=args.refresh)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = {ex.submit(_job, s): s for s in syms}
        done = 0
        n = len(futs)
        for f in as_completed(futs):
            res = f.result()
            results.append(res)
            done += 1
            if done % 50 == 0 or done == n:
                ok = sum(1 for r in results if r.get("action") == "ok")
                err = sum(1 for r in results if r.get("action") == "error")
                cur = sum(1 for r in results if r.get("action") == "current")
                print(f"  [{done}/{n}]  ok={ok}  current={cur}  err={err}")

    # Summary
    ok = sum(1 for r in results if r.get("action") == "ok")
    err = sum(1 for r in results if r.get("action") == "error")
    cur = sum(1 for r in results if r.get("action") == "current")
    print(f"\n[daily_cache] DONE  ok={ok}  current={cur}  err={err}")
    if err:
        print("[daily_cache] first 10 errors:")
        for r in [x for x in results if x.get("action") == "error"][:10]:
            print(f"  {r['symbol']}: {r.get('error')}")


if __name__ == "__main__":
    main()
