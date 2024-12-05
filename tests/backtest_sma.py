import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumibot.backtesting import YahooDataBacktesting
from datetime import datetime

# Add the root directory of the project to PYTHONPATH
from strategies.sma_momentum import SMAMomentumRefined

# Set up backtesting dates. Adjust as needed
start_date = datetime(2024, 1, 1)
end_date = datetime.today()

# Run and display results
results = SMAMomentumRefined.run_backtest(
    YahooDataBacktesting,
    backtesting_start=start_date,
    backtesting_end=end_date,
)
