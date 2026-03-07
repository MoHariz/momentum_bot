"""
main.py — Live / Paper trading entry point for HalalMomentumBot
Broker: Alpaca (paper by default)
Deploy: Render (Free or Starter tier, always-on)

Environment variables required (set in Render dashboard or .env):
  ALPACA_API_KEY       — from Alpaca paper dashboard
  ALPACA_API_SECRET    — from Alpaca paper dashboard
  ALPACA_IS_PAPER      — "true" for paper, "false" for live (default: true)
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from strategies.halal_momentum import HalalMomentumBot

# ── Alpaca config ────────────────────────────────────────────────────────────

IS_PAPER = os.getenv("ALPACA_IS_PAPER", "true").lower() != "false"

ALPACA_CONFIG = {
    "API_KEY":    os.getenv("ALPACA_API_KEY",    ""),
    "API_SECRET": os.getenv("ALPACA_API_SECRET", ""),
    "PAPER":      IS_PAPER,
}

# ── Guard: refuse to start if keys are missing ───────────────────────────────

if not ALPACA_CONFIG["API_KEY"] or not ALPACA_CONFIG["API_SECRET"]:
    raise EnvironmentError(
        "ALPACA_API_KEY and ALPACA_API_SECRET must be set as environment variables."
    )

# ── Boot ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = "PAPER" if IS_PAPER else "LIVE"
    print(f"[HalalMomentumBot] Starting in {mode} mode...")

    broker = Alpaca(ALPACA_CONFIG)

    strategy = HalalMomentumBot(
        broker=broker,
        name="HalalMomentumBot",
    )

    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()