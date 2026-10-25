# CLAUDE.md

Halal-screened daily momentum bot. Lumibot + Alpaca, deployed as a Render background worker. The live strategy is `HalalMomentumBot` in `strategies/halal_momentum.py`. `simple_momentum.py` and `sma_momentum.py` are older versions (v1/v2) that nothing imports; don't extend them.

## Strategy rules

The code is the source of truth. If the README disagrees, trust the code and point out the mismatch.

- **Cadence:** `sleeptime = "1D"`. The bot runs once per trading day at market open, using 120 daily bars (SMA100 needs at least 100).
- **Regime gate (SPY):** Bullish = SMA50 > SMA100 and RSI(14) > 50. Bearish = SMA50 < SMA100 or RSI < 45. Anything else is Neutral. Only Bullish allows new entries.
- **Entry (all must hold):** Bullish regime, SMA20 > SMA50, RSI(14) > 50, MACD(12,26,9) > signal, no existing position.
- **Exit (either):** SMA20 < SMA50 or RSI(14) < 45. Exits run in every regime and must never be blocked by any limit.
- **Ordering:** stocks are processed in order of 20-day return / 20-day volatility (`_rank_assets`).
- **Sizing:** 20% of portfolio value per position, capped by available cash minus cash already committed in this run, with a 1-share floor. Size from cash, never from buying power (see halal constraints).
- **Daily cap:** `max_daily_entries = 3` limits new buys only.
- **Circuit breaker:** at -15% from peak equity, close everything and set `halted`. It stays halted until the regime is Bullish, then resets the peak to current equity and resumes. While halted, it retries any failed closes.
- **Persisted state:** `portfolio_peak` and `halted` are saved to `STATE_FILE` as JSON with atomic writes. `_load_state` / `_save_state` skip this when `self.is_backtesting`, so backtests start fresh. Any new state that must survive restarts goes through these two methods.
- **Known open risks (deferred):** (a) held stocks that drop out of `_rank_assets()` skip their exit check; (b) there is no cap on open positions, so all 4 can be held despite the docstring's "max 3".

## Halal constraints (hard rules)

Treat these as non-negotiable. Don't add code that breaks them, even as an option or flag.

- Long-only equities. No short selling, options, futures, leveraged or inverse ETFs, or crypto derivatives.
- No margin or borrowing. Size orders from settled cash (`get_cash()`), not buying power, because Alpaca accounts can be margin-enabled.
- No interest-bearing instruments (bonds, bond ETFs, money-market sweeps used as a holding).
- The universe is currently `MSFT, GOOGL, TSLA, AAPL`, drawn from the HLAL/SPUS overlap. Only add tickers the user confirms are in current HLAL/SPUS holdings. Don't infer compliance yourself.
- Shariah screening status changes over time (debt ratios, revenue mix). If universe changes come up, remind the user to re-screen. SPY is used only as a regime signal and is never traded.

## Deployment gotchas (Render)

- **Python 3.11.9** is pinned in `.python-version`. Newer Pythons (3.14) break the PyYAML build, so keep `PyYAML==6.0.1` and `aiohttp<4` pinned in `requirements.txt`.
- **Build command:** it installs with uv into `.venv/bin/python`. Don't change it casually, because it took several fix commits to get working.
- **Persistent disk at `/data2026`:** Render's normal filesystem is wiped on every restart and deploy. `STATE_FILE=/data2026/bot_state.json` is set in `render.yaml`. If it's unset, the bot writes to `state/bot_state.json`, which is lost on restart, and nothing errors. Lumibot's SQLite (Discord integration) also needs this disk.
- **Healthy logs:** first boot shows `No state file at ... Starting fresh`. Later boots show `Loaded state from ...`. `WARNING: could not write state file` means the disk isn't attached or writable.
- **Paper vs live:** paper is the default. Only `ALPACA_IS_PAPER=false` (case-insensitive) switches to live. Never change this default or the `render.yaml` value without the user asking explicitly. `main.py` refuses to start without `ALPACA_API_KEY` / `ALPACA_API_SECRET`.
- **Secrets:** they live in `.env` locally (gitignored) and in the Render dashboard (`sync: false`). Never commit keys.

## Working in this repo

- `tests/` holds runnable scripts, not a pytest suite. There are no automated tests.
  - `python tests/backtest_yahoo.py`: Yahoo data, historical regimes back to 2018.
  - `python tests/backtest_sma.py`: Polygon data, needs `POLYGON_API_KEY`. Despite the name, it backtests `HalalMomentumBot`.
  - `python tests/alpaca_api_test.py`: checks the Alpaca connection.
- To test strategy logic without network calls, stub the broker methods on an instance (`get_portfolio_value`, `get_position`, `get_cash`, `get_last_price`, `_fetch_with_retry`). Note that `positions` is a read-only property on Lumibot's `Strategy`.
- Never run `main.py` with live keys.
- `gh` isn't installed. After pushing, give the user the GitHub compare URL and a PR description to paste.
- Branches follow the `itr-N` pattern off `main`.
