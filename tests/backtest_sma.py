import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumibot.backtesting import YahooDataBacktesting
from datetime import datetime

# Add the root directory of the project to PYTHONPATH
from strategies.sma_momentum import SMAMomentumBot
from strategies.simple_momentum import SimpleMomentumBot

# Define testing timelines
testing_timelines = {
    "Flat Market (2018 Trade War Volatility)": {
        "start_date": datetime(2018, 1, 1),
        "end_date": datetime(2019, 1, 1),
    },
    "Long Bull Market (2009–2019)": {
        "start_date": datetime(2009, 3, 1),
        "end_date": datetime(2019, 12, 1),
    },
    "Bear Market (COVID-19 Crash)": {
        "start_date": datetime(2020, 2, 1),
        "end_date": datetime(2020, 4, 1),
    },
    "Bull Market (COVID-19 Recovery)": {
        "start_date": datetime(2020, 4, 1),
        "end_date": datetime(2021, 12, 1),
    },
    "High Volatility (Post-Election Period)": {
        "start_date": datetime(2020, 11, 1),
        "end_date": datetime(2021, 1, 31),
    },
    "Inflationary Period (2022 Downturn)": {
        "start_date": datetime(2022, 1, 1),
        "end_date": datetime(2022, 12, 31),
    },
    "Tech Rally (2023 Recovery)": {
        "start_date": datetime(2023, 1, 1),
        "end_date": datetime(2023, 12, 31),
    },
    "Comprehensive Backtest": {
        "start_date": datetime(2009, 1, 1),
        "end_date": datetime.today(),
    },
    "YTD": {
        "start_date": datetime(datetime.now().year, 1, 1),
        "end_date": datetime.today(),
    },
}


# Run backtests for each timeline
for test_name, params in testing_timelines.items():
    print(f"Running backtest for: {test_name}")

    results = SimpleMomentumBot.run_backtest(
        YahooDataBacktesting,
        backtesting_start=params["start_date"],
        backtesting_end=params["end_date"],
        budget=5000,  # Adjust budget as needed
        name="SimpleMomentumBot during " + test_name,
        show_plot=True,
        show_tearsheet=True
    )
    print(f"Results for {test_name}:")
    print(results)
    print("\n")