# CLAUDE.md

HalalMomentumBot: a halal-screened daily momentum strategy on Lumibot + Alpaca, paper trading only, deployed as a Render background worker. **This repo is public.** Never write API keys, account balances, positions, or personal financial details into any file, commit, or PR.

## File map

- `strategies/halal_momentum.py`: `HalalMomentumBot`, the only live strategy
- `strategies/helper.py`: indicators (RSI, MACD, ATR, ADX, SMA slope). The live bot uses RSI and MACD only.
- `strategies/simple_momentum.py`, `strategies/sma_momentum.py`: legacy v1/v2, not imported anywhere. Don't extend them.
- `main.py`: entry point. Builds the Alpaca broker, refuses to start without keys, runs the strategy.
- `render.yaml`: Render worker config (build/start commands, env vars)
- `tests/`: runnable scripts, **not** a pytest suite. There are no automated tests.
  - `backtest_yahoo.py`: Yahoo data, the six regime periods (the regression baseline)
  - `backtest_sma.py`: Polygon data, 2025-01-01 to yesterday, needs `POLYGON_API_KEY`. Despite the name, it runs `HalalMomentumBot`.
  - `alpaca_api_test.py`: Alpaca connection check

## Strategy rules (as implemented)

Rules last changed in commit `938c520` (2026-10-25): the drawdown breaker now latches and resumes on a Bullish regime, and the daily cap limits entries only, so exits are never blocked.

- **Universe:** `MSFT, GOOGL, TSLA, AAPL`. SPY is fetched for the regime only and is never traded.
- **Cadence:** `sleeptime = "1D"`, one run per trading day.
- **Lookback:** 120 daily bars. **Must stay above 100.** SPY SMA100 returns NaN with fewer bars, and the regime then silently falls back to Neutral, which blocks all entries with no error.
- **SPY regime:** Bullish = SMA50 > SMA100 and RSI(14) > 50. Bearish = SMA50 < SMA100 or RSI < 45. Otherwise Neutral. A missing SPY fetch or any NaN also gives Neutral. Only Bullish allows entries.
- **Entry (all):** Bullish regime, SMA20 > SMA50, RSI(14) > 50, MACD(12,26,9) > signal, no current position.
- **Exit (either):** SMA20 < SMA50 or RSI(14) < 45. Checked in every regime and never blocked by the daily cap.
- **Processing order:** by 20-day return / 20-day volatility (`_rank_assets`). A stock missing from the ranking is skipped entirely, **including its exit check** (known open issue).
- **Sizing:** 20% of portfolio value, capped by `get_cash()` minus cash already committed this run, with a 1-share floor. There is **no limit on open positions**: the "max 3 positions" in the docstring isn't enforced, so all 4 can be held (~80%).
- **Daily cap:** `max_daily_entries = 3`, counting **buys only**. Exits are unlimited. Counters reset in `after_market_closes`.
- **Circuit breaker:** at -15% from peak equity, close everything and set `halted`. It stays halted (retrying failed closes) until the regime is Bullish, then resets the peak to current equity and resumes.
- **State:** `portfolio_peak` and `halted` persist as JSON to `STATE_FILE`, written atomically. Skipped when `self.is_backtesting`.

## Hard rules

1. **Paper trading only.** Don't add live-trading code, real keys, or anything that sets `ALPACA_IS_PAPER=false`. The existing `ALPACA_IS_PAPER` switch stays in place, because the user may go live after the December review. Don't remove it and don't use it.
2. **Entry and exit logic is frozen until the user's December review** (a paper-trading evaluation is running). This covers entry/exit conditions, regime thresholds, sizing, caps, and breaker behaviour. Ask before changing any of it, even "obvious fixes".
3. **Backtest every approved rule change** on all six periods in `tests/backtest_yahoo.py` before proposing it, and report results next to the baseline below.
4. **Halal constraints:** long-only equities. No options, leverage (margin or leveraged/inverse ETFs), short selling, or interest-bearing instruments. Size from cash, never buying power. **Universe changes need the user's approval.** Don't judge Shariah compliance yourself.
5. **Branches only.** Work on `itr-N` branches off `main`. **Never push to `main`**, because Render auto-deploys from it. Changes reach `main` through a PR the user merges.

## Deployment (Render)

- **Plan and deploys:** Render Starter (paid) background worker, auto-deploying on every push to `main` (the repo's default branch). Start command: `python main.py`.
- **Python 3.11.9** is pinned in `.python-version`. Newer Python (3.14) fails to build PyYAML, so `PyYAML==6.0.1` and `aiohttp<4` stay pinned too.
- **uv:** the build runs `pip install uv && uv pip install --python .venv/bin/python -r requirements.txt`. pip's resolver failed with `resolution-too-deep` on Lumibot's ccxt/coincurve dependency chain, so the build uses uv. Don't switch back to pip.
- **Persistent disk** at `/data2026`. Render's normal filesystem is wiped on every restart and deploy.
  - It was added in March 2026 for Lumibot's Discord summary. Lumibot creates SQLite tables (`strategy_tracker`, `vars_backup`).
  - **Unverified:** which path Lumibot's SQLite DB uses, so it may not actually be on the disk.
  - Breaker state lives at `STATE_FILE=/data2026/bot_state.json`. If it's unset, state goes to `state/bot_state.json` and is lost on restart, with no error.
- **Healthy logs:** first boot shows `No state file at ... Starting fresh`. Later boots show `Loaded state from ...`. `WARNING: could not write state file` means the disk isn't attached or writable.
- **Memory:** 512MB limit. Avoid loading more history or symbols than needed.
- **Env var names** (values only in the Render dashboard or a local `.env`, never in the repo): `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_IS_PAPER`, `STATE_FILE`, `DISCORD_WEBHOOK_URL`. `POLYGON_API_KEY` is for local backtests only.

## Backtesting

- Run `python tests/backtest_yahoo.py`. Only `2025` is active by default; uncomment the other five periods in `testing_timelines` to run all six.
- Settings: $5,000 budget, daily bars, Yahoo data. `show_plot`/`show_tearsheet` open browser windows. The script prints the regime distribution at the end.

Baseline (Yahoo, $5k). **Recorded before commit `938c520`** and not yet re-run. Re-run and replace these before treating them as a regression reference:

| Period | Dates | Return | Max DD | Sharpe | Notes |
|---|---|---|---|---|---|
| 2018 Q4 Selloff | 2018-10-01 → 2019-03-31 | -2.2% | -3.8% | -1.33 | 80% Bearish |
| 2020 COVID Crash + Recovery | 2020-02-01 → 2020-12-31 | +1.8% | -15.4% | 0.10 | DD past -15%; breaker now latches, so expect a change |
| 2021 Bull Run | 2021-01-01 → 2021-12-31 | +19.0% | -5.5% | 1.70 | 75% Bullish |
| 2022 Bear Market | 2022-01-01 → 2022-12-31 | 0.0% | 0.0% | — | 92% Bearish, zero trades |
| 2022–23 Transition | 2022-10-01 → 2023-06-30 | +5.7% | -5.3% | 0.26 | 45% Bullish |
| 2025 | 2025-01-01 → 2025-12-31 | +10.0% | -5.7% | 0.74 | 52% Bullish |

Signs of a regression: zero trades outside 2022 (often a lookback/NaN problem), or regime percentages that differ a lot from the baseline.

## Gotchas

- **Lookback/NaN fallback:** dropping the lookback below ~101 bars gives no error. The regime just reads Neutral every day.
- **Regime counter:** `_regime_counts` is class-level. `backtest_yahoo.py` reads it after the run, so don't make it an instance attribute without updating the script.
- **Deprecated pandas option:** `calculate_adx` uses `mode.use_inf_as_na`, which newer pandas removed. It's unused today, but it will crash if wired in.
- **RSI smoothing:** RSI uses a simple rolling mean, not Wilder smoothing, so values differ from broker charts.
- **Live switch:** `ALPACA_IS_PAPER` switches to live only for exactly `false` (case-insensitive). `tests/alpaca_api_test.py` passes the raw string as `PAPER`.
- **Unused dependency:** `alpaca-trade-api` is listed in `requirements.txt` but never imported. Lumibot uses `alpaca-py`.
- **dotenv not listed:** the test scripts import `dotenv`, but `python-dotenv` isn't in `requirements.txt`. Install it locally.
- **Stubbing Lumibot:** when stubbing `Strategy` in quick tests, note that `positions` is a read-only property.
- **No `gh` CLI:** `gh` isn't installed. After pushing, give the user the GitHub compare URL and a PR body to paste.
