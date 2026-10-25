"""
Fake-data check: a held stock that drops out of _rank_assets() must still get
its exit check, and entry behavior must not change.

No network or broker needed. Run from the repo root:
    python tests/unranked_exit_check.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from strategies.halal_momentum import HalalMomentumBot, MarketCondition


class FakePosition:
    def __init__(self, quantity):
        self.quantity = quantity


class FakeBot(HalalMomentumBot):
    """HalalMomentumBot with broker calls replaced by in-memory fakes."""

    @property
    def is_backtesting(self):
        return True  # skip STATE_FILE reads/writes

    def __init__(self, held):
        self.logs, self.buys, self.sells = [], [], []
        self._held = dict(held)
        self.initialize()

    def log_message(self, message, *args, **kwargs):
        self.logs.append(message)

    def get_position(self, stock):
        qty = self._held.get(stock, 0)
        return FakePosition(qty) if qty else None

    def get_cash(self):
        return 10_000

    def get_portfolio_value(self):
        return 10_000

    def get_last_price(self, stock):
        return 100.0

    def _place_buy(self, stock, quantity):
        self.buys.append(stock)
        self.day_trades_count["buy"] += 1

    def _close_position(self, stock):
        self.sells.append(stock)
        self._held.pop(stock, None)


def rising(n=120):
    return pd.DataFrame({"close": np.linspace(100, 200, n)})


def falling(n=120):
    return pd.DataFrame({"close": np.linspace(200, 100, n)})


def falling_unrankable():
    """Downtrend with a zero close 21 bars back: volatility is NaN, so
    _rank_assets() drops it, but SMA/RSI/MACD are still computable."""
    df = falling()
    df.loc[len(df) - 21, "close"] = 0.0
    return df


def run(held, cache, regime=MarketCondition.Bullish):
    bot = FakeBot(held)
    bot._price_cache = cache
    bot._run_momentum_strategy(regime)
    return bot


def check(name, condition):
    print(f"{'PASS' if condition else 'FAIL'}  {name}")
    return condition


if __name__ == "__main__":
    results = []

    # 1. Held + unranked + exit signal -> must be sold.
    bot = run({"TSLA": 5}, {"MSFT": rising(), "TSLA": falling_unrankable()})
    results.append(check("unranked held stock is ranked out (precondition)",
                         "TSLA" not in bot._rank_assets()))
    results.append(check("unranked held stock with exit signal is sold",
                         bot.sells == ["TSLA"]))

    # 2. Held + no data at all -> not sold, but now logged instead of silent.
    bot = run({"AAPL": 3}, {"MSFT": rising()})
    results.append(check("held stock with failed fetch is logged",
                         "No cached data for AAPL, skipping." in bot.logs))
    results.append(check("held stock with failed fetch is not sold", bot.sells == []))

    # 3. Entries unchanged: unranked *unheld* stock is never bought, ranked
    #    candidates are bought in rank order, and the daily cap still applies.
    cache = {s: rising() for s in ["MSFT", "GOOGL", "AAPL"]}
    cache["TSLA"] = falling_unrankable()
    bot = run({}, cache)
    results.append(check("unranked unheld stock is not bought", "TSLA" not in bot.buys))
    results.append(check("ranked candidates bought in rank order",
                         bot.buys == bot._rank_assets()[:bot.max_daily_entries]))

    # 4. Non-Bullish regime: unranked held stock still exits, nothing is bought.
    bot = run({"TSLA": 5}, {"MSFT": rising(), "TSLA": falling_unrankable()},
              regime=MarketCondition.Bearish)
    results.append(check("exit fires in Bearish regime too",
                         bot.sells == ["TSLA"] and bot.buys == []))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
