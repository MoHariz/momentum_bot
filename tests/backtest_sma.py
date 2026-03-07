import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumibot.backtesting import PolygonDataBacktesting
from datetime import datetime, timedelta, time as dt_time
from dotenv import load_dotenv

load_dotenv()

from strategies.halal_momentum import HalalMomentumBot

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY_HERE")

testing_timelines = {
    # "Bear Market (COVID-19 Crash)": {
    #     "start_date": datetime(2020, 2, 1),
    #     "end_date": datetime(2020, 4, 1),
    # },
    # "Bull Market (COVID-19 Recovery)": {
    #     "start_date": datetime(2020, 4, 1),
    #     "end_date": datetime(2021, 12, 1),
    # },
    # "Inflationary Period (2022 Downturn)": {
    #     "start_date": datetime(2022, 1, 1),
    #     "end_date": datetime(2022, 12, 31),
    # },
    # "Tech Rally (2023 Recovery)": {
    #     "start_date": datetime(2023, 1, 1),
    #     "end_date": datetime(2023, 12, 31),
    # },
    # "YTD": {
    #     "start_date": datetime(datetime.now().year, 1, 1),
    #     "end_date": datetime.today(),
    # },
    "2025": {
        "start_date": datetime(2025, 1, 1),
        "end_date": datetime.today() - timedelta(days=1),
    },
}

if __name__ == "__main__":
    for test_name, params in testing_timelines.items():
        print(f"\nRunning backtest: {test_name}")
        print(f"  {params['start_date'].date()} -> {params['end_date'].date()}")

        time.sleep(2)

        try:
            results = HalalMomentumBot.backtest(
                PolygonDataBacktesting,
                params["start_date"],
                params["end_date"],
                budget=5000,
                name=f"HalalMomentumBot - {test_name}",
                show_plot=True,
                show_tearsheet=True,
                benchmark_asset=None,
                polygon_api_key=POLYGON_API_KEY,
                polygon_has_paid_subscription=False,
            )
            results_tuple = results if isinstance(results, tuple) else (results,)
            stats = results_tuple[0]
            print(f"\nResults for {test_name}:")
            print(stats)

            # Find strategy instance in results tuple
            strategy_instance = None
            for item in results_tuple:
                if hasattr(item, '_regime_counts'):
                    strategy_instance = item
                    break

            if strategy_instance:
                counts = strategy_instance._regime_counts
                total = sum(counts.values())
                if total > 0:
                    print(f"\nRegime distribution ({total} trading days):")
                    for regime, count in counts.items():
                        print(f"  {regime:8s}: {count:4d} days ({count/total*100:.1f}%)")
            else:
                print(f"\nResults tuple had {len(results_tuple)} items: {[type(r).__name__ for r in results_tuple]}")

        except Exception as e:
            print(f"Backtest failed for {test_name}: {e}")
            time.sleep(10)