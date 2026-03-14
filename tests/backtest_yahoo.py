import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumibot.backtesting import YahooDataBacktesting
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from strategies.halal_momentum import HalalMomentumBot

testing_timelines = {
    "2025": {
        "start_date": datetime(2025, 1, 1),
        "end_date": datetime(2025, 12, 31),
    },
    # "2022 Bear Market": {
    #     "start_date": datetime(2022, 1, 1),
    #     "end_date": datetime(2022, 12, 31),
    # },
    # "2020 COVID Crash + Recovery": {
    #     "start_date": datetime(2020, 2, 1),
    #     "end_date": datetime(2020, 12, 31),
    # },
    # "2022-2023 Bear to Bull Transition": {
    #     "start_date": datetime(2022, 10, 1),
    #     "end_date": datetime(2023, 6, 30),
    # },
    # "2021 Bull Run": {
    #     "start_date": datetime(2021, 1, 1),
    #     "end_date": datetime(2021, 12, 31),
    # },
    # "2018 Q4 Selloff": {
    #     "start_date": datetime(2018, 10, 1),
    #     "end_date": datetime(2019, 3, 31),
    # },
}

if __name__ == "__main__":
    for test_name, params in testing_timelines.items():
        print(f"\nRunning backtest: {test_name}")
        print(f"  {params['start_date'].date()} -> {params['end_date'].date()}")

        time.sleep(2)

        try:
            results = HalalMomentumBot.backtest(
                YahooDataBacktesting,
                params["start_date"],
                params["end_date"],
                budget=5000,
                name=f"HalalMomentumBot - {test_name}",
                show_plot=True,
                show_tearsheet=True,
                benchmark_asset=None,
            )
            results_tuple = results if isinstance(results, tuple) else (results,)
            stats = results_tuple[0]
            print(f"\nResults for {test_name}:")
            print(stats)

            # Print regime distribution from class-level counter
            counts = HalalMomentumBot._regime_counts
            total = sum(counts.values())
            if total > 0:
                print(f"\nRegime distribution ({total} trading days):")
                for regime, count in counts.items():
                    print(f"  {regime:8s}: {count:4d} days ({count/total*100:.1f}%)")
            else:
                print("\nNo regime data recorded.")

        except Exception as e:
            print(f"Backtest failed for {test_name}: {e}")
            time.sleep(10)