"""
alpaca_auth.py
==============
Alpaca authentication & client factory.  Mirror of `Kiteconnect.py` from
the Indian port, restructured for the alpaca-py SDK.

USAGE
-----
    from alpaca_auth import get_data_client, get_trading_client

    data = get_data_client()                 # historical bars
    trade = get_trading_client(paper=True)   # orders / positions

ENVIRONMENT VARIABLES
---------------------
    ALPACA_API_KEY         required
    ALPACA_SECRET_KEY      required
    ALPACA_TOKEN_FILE      optional, JSON: {"api_key":"...","secret_key":"..."}
    ALPACA_PAPER           "1"/"true" -> default to paper trading endpoint
    ALPACA_DATA_FEED       "iex" (default, free) | "sip" (paid subscription)

NOTES
-----
The Indian original needed a manual access-token refresh dance (Kite uses
a daily-rotating token that requires browser login). Alpaca uses static
API keys, so there's no equivalent of `Kiteconnect.py`'s GUI prompt; just
read the token file or env vars at startup.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# alpaca-py is loaded lazily so this module can be imported on systems
# that haven't installed the SDK yet (eg unit tests of cache modules with
# mocked clients).
_ALPACA_IMPORT_ERR: Optional[str] = None
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed, Adjustment
    from alpaca.trading.client import TradingClient
    _ALPACA_OK = True
except Exception as e:  # pragma: no cover - import-time guard
    _ALPACA_OK = False
    _ALPACA_IMPORT_ERR = str(e)


class AuthExpired(Exception):
    """Alpaca API keys missing or invalid (mirrors the Indian original)."""


@dataclass(frozen=True)
class AlpacaCreds:
    api_key: str
    secret_key: str
    paper: bool = True
    data_feed: str = "iex"  # "iex" (free) or "sip" (paid)


# ---------------------------------------------------------------------------
# Token / credential discovery
# ---------------------------------------------------------------------------

def _token_file_path() -> Optional[Path]:
    env = os.environ.get("ALPACA_TOKEN_FILE")
    if env:
        return Path(env)
    candidates = [
        Path.home() / ".alpaca" / "token.json",
        Path.cwd() / "alpaca_token.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _read_token_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def load_creds() -> AlpacaCreds:
    """
    Resolution order:
      1. ALPACA_API_KEY / ALPACA_SECRET_KEY env vars
      2. JSON file at ALPACA_TOKEN_FILE
      3. JSON file at ~/.alpaca/token.json or ./alpaca_token.json
    """
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not (api_key and secret_key):
        tp = _token_file_path()
        if tp is None:
            raise AuthExpired(
                "Alpaca credentials missing. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY, or write {api_key, secret_key} JSON "
                "to ~/.alpaca/token.json or ./alpaca_token.json."
            )
        try:
            data = _read_token_file(tp)
        except Exception as e:
            raise AuthExpired(f"Failed to read {tp}: {e}") from e
        api_key = api_key or data.get("api_key") or data.get("ALPACA_API_KEY")
        secret_key = secret_key or data.get("secret_key") or data.get("ALPACA_SECRET_KEY")

    if not (api_key and secret_key):
        raise AuthExpired(
            "Alpaca credentials missing. api_key/secret_key not found in "
            "env or token file."
        )

    paper = _bool_env("ALPACA_PAPER", default=True)
    data_feed = (os.environ.get("ALPACA_DATA_FEED") or "iex").strip().lower()
    if data_feed not in ("iex", "sip"):
        print(f"[alpaca_auth] unknown ALPACA_DATA_FEED={data_feed!r}, "
              f"falling back to iex", file=sys.stderr)
        data_feed = "iex"

    return AlpacaCreds(api_key=api_key, secret_key=secret_key,
                        paper=paper, data_feed=data_feed)


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

def _check_sdk():
    if not _ALPACA_OK:
        raise RuntimeError(
            "alpaca-py is not installed. Run `pip install alpaca-py>=0.30`. "
            f"Original import error: {_ALPACA_IMPORT_ERR}"
        )


def get_data_client(creds: Optional[AlpacaCreds] = None):
    """Historical bars client. Used by daily_cache / intraday_cache /
    global_cache."""
    _check_sdk()
    if creds is None:
        creds = load_creds()
    return StockHistoricalDataClient(api_key=creds.api_key,
                                      secret_key=creds.secret_key)


def get_trading_client(paper: Optional[bool] = None,
                        creds: Optional[AlpacaCreds] = None):
    """Trading client (positions, orders, account). Defaults to paper."""
    _check_sdk()
    if creds is None:
        creds = load_creds()
    if paper is None:
        paper = creds.paper
    return TradingClient(api_key=creds.api_key,
                          secret_key=creds.secret_key,
                          paper=paper)


def resolved_data_feed(creds: Optional[AlpacaCreds] = None):
    """Returns the alpaca-py DataFeed enum corresponding to the user's
    subscription tier."""
    _check_sdk()
    if creds is None:
        creds = load_creds()
    return DataFeed.IEX if creds.data_feed == "iex" else DataFeed.SIP


def smoke_test() -> dict:
    """
    Loads creds, instantiates trading client, fetches account info, and
    pulls a single AAPL daily bar to confirm the data feed works.
    """
    creds = load_creds()
    out = {
        "api_key_present": bool(creds.api_key),
        "secret_key_present": bool(creds.secret_key),
        "paper": creds.paper,
        "data_feed": creds.data_feed,
    }
    try:
        tc = get_trading_client(creds=creds)
        acct = tc.get_account()
        out.update({
            "account_status": str(acct.status),
            "buying_power": float(getattr(acct, "buying_power", 0.0)),
            "equity": float(getattr(acct, "equity", 0.0)),
        })
    except Exception as e:
        out["trading_client_error"] = str(e)
    try:
        from datetime import datetime, timedelta, timezone
        dc = get_data_client(creds=creds)
        end = datetime.now(timezone.utc) - timedelta(days=2)
        start = end - timedelta(days=1)
        req = StockBarsRequest(
            symbol_or_symbols=["AAPL"],
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start, end=end,
            feed=resolved_data_feed(creds),
            adjustment=Adjustment.SPLIT,
        )
        bars = dc.get_stock_bars(req)
        out["data_client_sample_rows"] = int(len(bars.df) if bars and bars.df is not None else 0)
    except Exception as e:
        out["data_client_error"] = str(e)
    return out


def main():
    res = smoke_test()
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
