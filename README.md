# HalalMomentumBot

## Overview

HalalMomentumBot is a halal-screened momentum trading strategy built with Python, Lumibot, and Alpaca. It trades a curated universe of HLAL/SPUS ETF-overlapping stocks using a combination of SMA crossovers, RSI, and MACD signals, gated by a SPY-based market regime filter to avoid entering positions during downtrends.

The same strategy code runs both in backtesting and live paper trading, deployed on Render as a background worker.

### Version History

| Version | Class | Description |
|---|---|---|
| v1 | `SMAMomentumBot` | Basic SMA crossover strategy, no halal screening |
| v2 | `SimpleMomentumBot` | Added RSI and ADX filters, configurable universe |
| v3 | `HalalMomentumBot` | Halal-screened universe, MACD entry gate, SPY regime filter, Render deployment |

## Strategy

### Universe
`MSFT`, `GOOGL`, `TSLA`, `AAPL` — selected from HLAL/SPUS halal ETF overlap for liquidity and momentum characteristics.

### Entry Logic (all conditions must be true)
1. **Market regime is Bullish** — SPY SMA50 > SMA100 AND SPY RSI > 50
2. **SMA20 > SMA50** — stock is in an uptrend
3. **RSI(14) > 50** — momentum is positive
4. **MACD > Signal line** — momentum is accelerating
5. No existing position in the stock

### Exit Logic (either condition triggers)
- SMA20 < SMA50 (trend reversal)
- RSI(14) < 45 (momentum loss)

### Risk Management
- 20% of portfolio per position (max 3 positions = 60% invested)
- 15% max drawdown circuit breaker — closes all positions
- Max 3 trades per day
- ATR-based position sizing with 1-share floor

### Market Regime Filter
SPY is evaluated each iteration using SMA50/SMA100 + RSI:
- **Bullish** — entries allowed
- **Neutral / Bearish** — new entries blocked, existing exits still monitored

## Backtest Results

Backtested across multiple market regimes using Yahoo Finance data (daily bars, $5,000 starting capital):

| Period | Return | Max Drawdown | Sharpe | Notes |
|---|---|---|---|---|
| 2018 Q4 Selloff | -2.2% | -3.8% | -1.33 | 80% Bearish regime |
| 2020 COVID Crash + Recovery | +1.8% | -15.4% | 0.10 | Fast crash, SMA lag |
| 2021 Bull Run | +19.0% | -5.5% | 1.70 | 75% Bullish regime |
| 2022 Bear Market | 0.0% | 0.0% | — | 92% Bearish, zero trades |
| 2022–2023 Transition | +5.7% | -5.3% | 0.26 | 45% Bullish recovery |
| 2025 (full year) | +10.0% | -5.7% | 0.74 | 52% Bullish |

## Project Structure

```
momentum_bot/
├── strategies/
│   ├── halal_momentum.py     # Main strategy (entry/exit/regime logic)
│   └── helper.py             # RSI, MACD, ATR, SMA slope calculations
├── tests/
│   ├── backtest_sma.py       # Polygon-based backtester (live data, 2yr history)
│   └── backtest_yahoo.py     # Yahoo Finance backtester (historical regimes)
├── main.py                   # Live trading entry point (Render/Alpaca)
├── render.yaml               # Render deployment config
├── requirements.txt          # Python dependencies
└── .python-version           # Pinned to 3.11.9 for Render compatibility
```

## Setup and Installation

### 1. Clone the Repository

```sh
git clone <repository-url>
cd momentum_bot
```

### 2. Create Conda Environment

```sh
conda create -n momentum-bot python=3.10
conda activate momentum-bot
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory:

```ini
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
ALPACA_IS_PAPER=True
POLYGON_API_KEY=your_polygon_api_key
```

## Usage

### Backtesting (Polygon — recent data)

```sh
python tests/backtest_sma.py
```

Edit `testing_timelines` in the script to select date ranges.

### Backtesting (Yahoo Finance — historical regimes)

```sh
python tests/backtest_yahoo.py
```

Supports date ranges back to 2018. Uncomment the desired periods in `testing_timelines`.

### Live / Paper Trading

```sh
python main.py
```

Runs against Alpaca paper trading by default (`ALPACA_IS_PAPER=True`). The bot fires once per trading day at market open (13:31 UTC), logs signals for all 4 symbols, and sends a daily summary to Discord.

## Deployment (Render)

The bot runs as a background worker on Render's free tier (512MB RAM).

Key deployment files:
- `render.yaml` — worker type, build/start commands
- `.python-version` — pinned to `3.11.9` to avoid PyYAML build failures on Python 3.14
- A persistent Render disk is required for Lumibot's SQLite-based Discord integration

```sh
# Build command
pip install uv && uv pip install --python .venv/bin/python -r requirements.txt

# Start command
python main.py
```

Required Render environment variables:
```
ALPACA_API_KEY
ALPACA_API_SECRET
ALPACA_IS_PAPER=true
DISCORD_WEBHOOK_URL (set in Lumibot strategy config)
```

## Discord Integration

The bot sends a daily account summary to a Discord channel via Lumibot's native webhook integration, including portfolio value, open positions, and a running equity chart.

## Halal Compliance

- Universe screened against HLAL and SPUS ETF holdings
- No options, leverage, or short selling
- No interest-bearing instruments
- Pure equity ownership in halal-screened companies

## License

MIT License

## Deploy to Render
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)