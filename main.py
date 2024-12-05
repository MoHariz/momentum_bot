import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from strategies.sma_momentum import SMAMomentumBot

import dotenv

dotenv.load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_API_SECRET")

ALPACA_CONFIG = {
    # Put your own Alpaca key here:
    "API_KEY": api_key,
    # Put your own Alpaca secret here:
    "API_SECRET": api_secret,
    "PAPER": True
}

trader = Trader()
broker = Alpaca(ALPACA_CONFIG)
strategy = SMAMomentumBot(broker=broker)

# Run the strategy live
trader.add_strategy(strategy)
trader.run_all()