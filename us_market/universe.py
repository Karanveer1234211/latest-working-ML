"""
universe.py
===========
S&P 500 + Russell 1000 universe builder for the US port.

The Indian original loaded its universe from a user-supplied CSV/text file
(via `_load_symbols_from_file`). We do the same here, but provide a default
fetcher for clean public sources:

  - S&P 500 constituents from Wikipedia (always available, free, daily-fresh)
  - Russell 1000 constituents from Wikipedia (best-effort)
  - Optional: Alpaca's full equity tradable list, filtered to NYSE/NASDAQ
    common stock that is fractionable

USAGE
-----
    # Refresh the universe file
    python universe.py --rebuild

    # Read it from another script
    from universe import load_universe
    syms = load_universe()  # list[str]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#
# All US data lives INSIDE the us_market/ package by default. The Indian
# pipeline writes to its own paths (C:\Users\karanvsi\Desktop\Kite Connect\...
# or whatever the user configured); the US port writes to us_market/data/.
# Override via env var US_CACHE_DIR if you want a different root.

PACKAGE_DIR = Path(__file__).resolve().parent      # .../us_market
DEFAULT_CACHE_DIR = Path(os.environ.get("US_CACHE_DIR",
                          str(PACKAGE_DIR / "data"))).expanduser()
DEFAULT_UNIVERSE_FILE = DEFAULT_CACHE_DIR / "universe.txt"
DEFAULT_UNIVERSE_META = DEFAULT_CACHE_DIR / "universe_meta.json"

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_R1000_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"


# ---------------------------------------------------------------------------
# Symbol sanitisation (mirrors Indian sanitize_symbol)
# ---------------------------------------------------------------------------

_VALID_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,7}$")
_HEADER_WORDS = {"symbol", "symbols", "ticker", "tickers", "scrip", "name"}


def sanitize_symbol(raw: object) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    s = s.replace("\x00", "").replace("\r", " ").replace("\t", " ").replace("\n", " ")
    s = "".join(s.split())
    if not s or s.casefold() in _HEADER_WORDS:
        return None
    if not _VALID_TICKER.match(s):
        return None
    return s


def filename_safe(symbol: str) -> str:
    """Convert a market symbol to a filesystem-safe filename stub.
    BRK.B -> BRK_B (parquet writers dislike `.` in stems on some FS)."""
    return symbol.replace(".", "_")


# ---------------------------------------------------------------------------
# Constituent fetchers
# ---------------------------------------------------------------------------

def fetch_sp500_from_wikipedia() -> List[str]:
    """S&P 500 constituents from Wikipedia. Stable, no auth needed."""
    print(f"[universe] fetching S&P 500 from Wikipedia ...")
    try:
        tables = pd.read_html(WIKIPEDIA_SP500_URL, match="Symbol")
    except Exception as e:
        raise SystemExit(f"FATAL: could not fetch Wikipedia S&P 500 list: {e}")
    if not tables:
        raise SystemExit("FATAL: no Symbol table found on Wikipedia S&P 500 page")
    df = tables[0]
    sym_col = next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), None)
    if sym_col is None:
        raise SystemExit(f"FATAL: no symbol column found, got: {df.columns.tolist()}")
    syms = []
    for raw in df[sym_col]:
        s = sanitize_symbol(raw)
        if s:
            syms.append(s)
    syms = sorted(set(syms))
    print(f"[universe]   S&P 500: {len(syms)} symbols")
    return syms


def fetch_russell1000_from_wikipedia() -> List[str]:
    """Russell 1000 constituents from Wikipedia. Less reliable."""
    try:
        print(f"[universe] fetching Russell 1000 from Wikipedia ...")
        tables = pd.read_html(WIKIPEDIA_R1000_URL, match="Ticker")
        if not tables:
            return []
        df = tables[0]
        sym_col = next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), None)
        if sym_col is None:
            return []
        syms = sorted({sanitize_symbol(s) for s in df[sym_col] if sanitize_symbol(s) is not None})
        print(f"[universe]   Russell 1000: {len(syms)} symbols")
        return syms
    except Exception as e:
        print(f"[universe]   warning: Russell 1000 fetch failed: {e}")
        return []


def fetch_alpaca_tradable_assets() -> List[str]:
    """Optional: pull the full tradable equity universe directly from Alpaca."""
    try:
        from alpaca_auth import get_trading_client
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
    except Exception as e:
        print(f"[universe] alpaca-py not available, skipping Alpaca universe: {e}")
        return []
    try:
        tc = get_trading_client()
        req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        assets = tc.get_all_assets(req)
    except Exception as e:
        print(f"[universe] Alpaca universe fetch failed: {e}")
        return []
    syms: Set[str] = set()
    for a in assets:
        if not getattr(a, "tradable", True):
            continue
        if not getattr(a, "fractionable", False):
            continue
        if str(getattr(a, "exchange", "")).upper() not in ("NYSE", "NASDAQ"):
            continue
        s = sanitize_symbol(a.symbol)
        if s:
            syms.add(s)
    out = sorted(syms)
    print(f"[universe]   Alpaca tradable: {len(out)} symbols")
    return out


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def build_universe(use_alpaca: bool = False) -> List[str]:
    sp500 = fetch_sp500_from_wikipedia()
    r1000 = fetch_russell1000_from_wikipedia()
    alpaca = fetch_alpaca_tradable_assets() if use_alpaca else []

    union: Set[str] = set(sp500) | set(r1000)
    if alpaca:
        before = len(union)
        union = union & set(alpaca)
        print(f"[universe] intersected with Alpaca tradable: "
              f"{before} -> {len(union)} symbols")
    out = sorted(union)
    print(f"[universe] final universe size: {len(out)}")
    return out


def write_universe_file(symbols: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in symbols:
            f.write(s + "\n")
    print(f"[universe] wrote {len(symbols)} symbols -> {path}")


def write_universe_meta(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def load_universe(path: Optional[Path] = None) -> List[str]:
    """Read the universe file produced by build_universe()."""
    p = Path(path) if path else DEFAULT_UNIVERSE_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"Universe file not found: {p}. "
            f"Run `python universe.py --rebuild` first."
        )
    with open(p, "r", encoding="utf-8") as f:
        syms = [sanitize_symbol(line) for line in f]
    syms = [s for s in syms if s]
    return sorted(set(syms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="re-fetch from sources, overwrite universe file")
    ap.add_argument("--use-alpaca", action="store_true",
                    help="intersect with Alpaca's tradable list (requires alpaca-py + creds)")
    ap.add_argument("--out-file", type=str, default=str(DEFAULT_UNIVERSE_FILE))
    args = ap.parse_args()

    out_file = Path(args.out_file)
    if out_file.exists() and not args.rebuild:
        syms = load_universe(out_file)
        print(f"[universe] reusing cached file ({len(syms)} symbols): {out_file}")
        return

    syms = build_universe(use_alpaca=args.use_alpaca)
    write_universe_file(syms, out_file)
    write_universe_meta({
        "n_symbols": len(syms),
        "built_at": pd.Timestamp.utcnow().isoformat(),
        "sources": ["wikipedia_sp500", "wikipedia_r1000",
                    *(["alpaca_tradable"] if args.use_alpaca else [])],
        "out_file": str(out_file),
    }, DEFAULT_UNIVERSE_META)


if __name__ == "__main__":
    main()
