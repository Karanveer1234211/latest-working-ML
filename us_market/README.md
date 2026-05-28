# US Market Port

Faithful port of the Indian-market trading model to US equities. Same
architecture, same indicators, same model pipeline; the data layer is
swapped from Zerodha Kite Connect to Alpaca.

## File mapping

| Indian original                  | US replica                  | Status |
| -------------------------------- | --------------------------- | ------ |
| `Kiteconnect.py`                 | `alpaca_auth.py`            | done   |
| `Daily cache.py`                 | `daily_cache.py`            | done   |
| `latest intraday cache.py`       | `intraday_cache.py`         | done   |
| `Global_cache.py`                | `global_cache.py`           | done   |
| _new_                            | `universe.py`               | done   |
| `Auto features.py`               | `auto_features.py`          | next PR |
| `New_model.py`                   | `new_model.py`              | next PR |
| `NEW FEAT IMP.py`                | `feat_importance.py`        | next PR |
| `compare.py`                     | `compare.py`                | next PR |
| `filt backtest.py`               | `filt_backtest.py`          | next PR |
| `ORB execution.py`               | `orb_execution.py`          | next PR |

## Market parameters (vs Indian original)

| Parameter            | Indian              | US                              |
| -------------------- | ------------------- | ------------------------------- |
| Exchanges            | NSE, BSE            | NYSE, NASDAQ, AMEX              |
| Timezone             | Asia/Kolkata        | America/New_York (DST-aware)    |
| Regular session      | 09:15-15:30 IST     | 09:30-16:00 ET                  |
| First 15 min         | 09:15-09:30         | 09:30-09:45                     |
| Macro proxy          | NIFTY 50, INDIA VIX | SPY (S&P 500), ^VIX (CBOE)      |
| Universe size        | ~1,500-2,000        | ~1,500 (S&P 500 + R1000 dedup)  |
| Liquidity filter     | close >= 2.0, avg20_vol >= 200K | close >= 5.0, avg20_vol >= 500K |
| Holiday calendar     | NSE                 | NYSE (`pandas_market_calendars`)|

## Quick start

```bash
pip install -r requirements.txt
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
# Optional: override the default cache dir.
# Default is us_market/data/ (lives inside this package).
# export US_CACHE_DIR=/some/other/path

# 1. Build/refresh universe (S&P 500 + Russell 1000, deduped)
python universe.py --rebuild

# 2. Backfill macro (SPY + VIX)
python global_cache.py --years 8

# 3. Backfill daily OHLCV + indicators for the universe
python daily_cache.py --years 6

# 4. Backfill intraday 5-min for last 2 years
python intraday_cache.py --years 2
```

## Isolation guarantees

The US code is fully scoped to `us_market/`:

  * **No edits to any Indian file.** Verified by `git diff main..us-market-foundation`:
    every changed path is under `us_market/`.
  * **All data goes under `us_market/data/`** by default:
    ```
    us_market/data/
    ├── universe.txt                       (S&P 500 + R1000 list)
    ├── universe_meta.json
    ├── macro_cache.parquet                (SPY + VIX)
    ├── daily/                             (one parquet per symbol + .ok.json)
    │   ├── AAPL_daily.parquet
    │   ├── AAPL_daily.ok.json
    │   └── ...
    └── intraday_5min/                     (one parquet per symbol + .ok.json)
        ├── AAPL.parquet
        ├── AAPL.ok.json
        └── ...
    ```
  * **All env vars are `US_*` prefixed** (`US_CACHE_DIR`, `US_DAILY_ROOT`,
    `US_INTRADAY_ROOT`, `US_GLOBAL_PATH`, `US_VIX_SOURCE`) so they cannot
    collide with the Indian pipeline's `CACHE_BASE_DIR`, `KITE_TOKEN_FILE`,
    etc.
  * **`us_market/data/` is gitignored** so cache artefacts never show up in
    PRs.

The Indian pipeline can keep running against its own paths (e.g.
`C:\Users\karanvsi\Desktop\Kite Connect\...`) with zero changes.

## Decisions worth knowing

1. **Why Alpaca, not Polygon/IBKR?**  Alpaca is the closest analog to Kite
   Connect: free historical bars, broker+data combo, paper & live trading
   in one SDK. Polygon is more accurate but $29-$199/month minimum.
   Switching providers later requires only changes inside `alpaca_auth.py`
   and the four `*_cache.py` modules.

2. **Free tier caveats.** The Alpaca free tier serves IEX-only data, not
   the full SIP feed. For research/backtest this is fine. For production
   ($99/mo) you get the full SIP feed.

3. **Schema version `v100`.** Bumped to avoid collision with the Indian
   `v19`. The on-disk parquet schema is otherwise identical, so the model
   pipeline works without changes.

4. **No GUI.** The Indian `Daily cache.py` had tkinter file-pickers; that
   was Windows-specific UX. The US port is CLI-only.

5. **Symbol normalization.** US tickers can have `.` (e.g. `BRK.B`) and
   `-` separators in some feeds. We sanitize to a canonical form (`BRK.B`
   for the API, `BRK_B` for filenames).

## Run-order dependency

```
                  alpaca_auth.py
                       |
       ___________________________________
      /              |              \     \
universe.py    daily_cache.py  intraday_cache.py  global_cache.py
      \              |              /     /
       \_____________v_____________/     /
              new_model.py  <-----------+
                |
       feat_importance.py
                |
         filt_backtest.py
                |
         orb_execution.py
```

The four cache modules can run in parallel once auth + universe are set up.
