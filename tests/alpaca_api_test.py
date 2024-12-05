from alpaca.trading.client import TradingClient
from lumibot.brokers import Alpaca
import os
import dotenv

dotenv.load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_API_SECRET")
is_paper = os.getenv("ALPACA_IS_PAPER")

# Test connection
try:
    trading_client = TradingClient(api_key, api_secret, paper=True)
    account = trading_client.get_account()
    print("Connection successful:", account)
except Exception as e:
    print("Error:", e)

ALPACA_CONFIG = {
    # Put your own Alpaca key here:
    "API_KEY": api_key,
    # Put your own Alpaca secret here:
    "API_SECRET": api_secret,
    "PAPER": is_paper
}

# Test connection of lumibot with Alpaca
try:
    alpaca = Alpaca(ALPACA_CONFIG)
    print(alpaca.get_time_to_open())
    print(alpaca.get_time_to_close())
    print(alpaca.is_market_open())
    print("Lumibot and Alpaca connection successful")
except Exception as e:
    print("Error:", e)
