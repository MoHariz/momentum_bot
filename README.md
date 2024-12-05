# SMA Momentum Trading Bot

## Overview

The SMA Momentum Trading Bot is a Python-based trading strategy leveraging simple moving averages (SMA) and momentum indicators to make trading decisions. It utilizes Alpaca for brokerage services and Lumibot for backtesting and live trading.

## Project Structure

- **strategies/sma_momentum.py**: Contains the main `SMAMomentumRefined` class implementing the trading logic.
- **tests/backtest_sma.py**: Script to backtest the SMA momentum strategy using historical data.
- **tests/alpaca_api_test.py**: Tests the connection with Alpaca API.
- **main.py**: Script to run the strategy live using Alpaca as the broker.

## Setup and Installation

1. **Clone the Repository**

   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**

   Ensure you have Python installed. Then, install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

3. **Environment Variables**

   Create a `.env` file in the root directory and add your Alpaca API credentials:

   ```ini
   ALPACA_API_KEY=your_api_key
   ALPACA_API_SECRET=your_api_secret
   ALPACA_IS_PAPER=True
   ```

## Usage

### Backtesting

To backtest the strategy, run the backtest script:

```sh
python tests/backtest_sma.py
```

### Live Trading

To run the strategy live, execute:

```sh
python main.py
```

## Key Features

- **Multiple Indicators**: Uses SMA, RSI, ADX, and ATR for trading signals.
- **Risk Management**: Incorporates drawdown control and risk per trade management.
- **Asset Universe**: Configurable asset universe based on initial capital.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Deploy to Render
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)
