from lumibot.strategies.strategy import Strategy
from lumibot.entities.asset import Asset
import pandas as pd
import time
from enum import Enum
from strategies.helper import calculate_atr, calculate_rsi, calculate_macd, calculate_sma_slope


class MarketCondition(Enum):
    """Market regime based on SPY SMA50/100 + RSI."""
    Bearish = "Bearish"
    Bullish = "Bullish"
    Neutral = "Neutral"


class HalalMomentumBot(Strategy):
    """
    Halal-screened momentum strategy.

    Universe: MSFT, GOOGL, TSLA, AAPL (HLAL/SPUS ETF overlap)

    Entry logic (per stock):
      - Market regime must be Bullish (SPY SMA50 > SMA100 AND RSI > 50)
      - SMA20 > SMA50 (trend confirmation)
      - RSI(14) > 50 (momentum confirmation)
      - MACD > signal line (momentum accelerating)
      - No existing position

    Exit logic (per stock):
      - SMA20 < SMA50 (trend reversal)
      - OR RSI(14) < 45 (momentum loss)
      - OR drawdown circuit breaker fires (closes all)

    Position sizing:
      - Fixed 20% of portfolio per position (max 3 positions = 60% invested)
      - Remaining 40% held as cash buffer
      - Intra-iteration cash tracking prevents over-allocation

    Risk management:
      - Max drawdown -15% closes all positions
      - Max 3 trades per day
    """

    # Class-level regime counter so it's accessible after backtest completes
    _regime_counts = {"Bullish": 0, "Neutral": 0, "Bearish": 0}

    def initialize(self):
        self.sleeptime = "1D"

        # Halal universe — 4 cleanest momentum stocks from HLAL/SPUS overlap
        self.universe = ["MSFT", "GOOGL", "TSLA", "AAPL"]

        # Entry/exit SMA periods
        self.sma_fast = 20
        self.sma_slow = 50

        # RSI parameters
        self.rsi_period = 14
        self.rsi_entry  = 50    # RSI must be above this to enter
        self.rsi_exit   = 45    # RSI below this triggers exit

        # Position sizing — fixed % of portfolio per position
        self.position_size_pct = 0.20   # 20% per stock, max 3 = 60% invested

        # Circuit breakers
        self.max_drawdown_pct = -15.0
        self.max_daily_trades = 3

        # State
        self.portfolio_peak = 0
        self.day_trades_count = {"buy": 0, "sell": 0}
        self._cash_committed = 0

        # Price cache — populated once per iteration
        self._price_cache = {}

        # Fetch settings
        self._fetch_retries = 3
        self._fetch_retry_wait = 5
        self._fetch_delay = 0

        # Lookback — 120 daily bars. SMA100 needs 100 bars minimum + buffer.
        self.lookback = 120
        self.lookback_timestep = "day"

        # Reset class-level regime counter for each new backtest run
        HalalMomentumBot._regime_counts = {"Bullish": 0, "Neutral": 0, "Bearish": 0}

    # -------------------------------------------------------------------------
    # Lifecycle hooks
    # -------------------------------------------------------------------------

    def before_market_opens(self):
        self.log_message("=== Before Market Opens ===")
        self.log_message(f"Portfolio value : {self.get_portfolio_value():.2f}")
        self.log_message(f"Cash            : {self.get_cash():.2f}")

    def on_trading_iteration(self):
        # Reset intra-iteration cash tracker
        self._cash_committed = 0

        # Build price cache once per iteration
        self._price_cache = {}
        for symbol in self.universe + ["SPY"]:
            df = self._fetch_with_retry(symbol)
            if df is not None:
                self._price_cache[symbol] = df

        # Determine market regime — now used as trade gate
        market_condition = self.determine_market_condition()
        HalalMomentumBot._regime_counts[market_condition.value] += 1

        drawdown = self.calculate_drawdown()
        self.log_message(f"Market: {market_condition.value} | Drawdown: {drawdown:.2f}%")

        # Circuit breaker — close everything and stop
        if drawdown < self.max_drawdown_pct:
            self.log_message(f"Drawdown {drawdown:.2f}% hit threshold. Closing all positions.")
            for stock in self.universe:
                position = self.get_position(stock)
                if position and position.quantity > 0:
                    self._close_position(stock)
            return

        self._run_momentum_strategy(market_condition)

    def after_market_closes(self):
        self.log_message("=== After Market Closes ===")
        self.log_message(f"Portfolio value : {self.get_portfolio_value():.2f}")
        self.log_message(f"Cash            : {self.get_cash():.2f}")
        self.log_message(
            f"Trades today — Buy: {self.day_trades_count['buy']} | Sell: {self.day_trades_count['sell']}"
        )
        self._reset_daily_state()

    def on_finish(self):
        """Log regime distribution at end of backtest."""
        counts = HalalMomentumBot._regime_counts
        total = sum(counts.values())
        if total > 0:
            self.log_message("=== Regime Distribution ===")
            for regime, count in counts.items():
                self.log_message(f"  {regime:8s}: {count:4d} days ({count/total*100:.1f}%)")

    def on_bot_crash(self, error):
        self.log_message(f"Bot crashed: {error}")
        for stock in self.universe:
            position = self.get_position(stock)
            if position and position.quantity > 0:
                self.log_message(f"Open position on crash: {stock} x{position.quantity}")

    # -------------------------------------------------------------------------
    # Data fetching
    # -------------------------------------------------------------------------

    def _fetch_with_retry(self, symbol, length=None):
        """Fetch historical daily bars with retry on rate limit errors."""
        length = length or self.lookback
        time.sleep(self._fetch_delay)
        for attempt in range(self._fetch_retries):
            try:
                bars = self.get_historical_prices(
                    symbol, length=length, timestep=self.lookback_timestep
                )
                if bars is not None and not bars.df.empty:
                    return bars.df
                return None
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str:
                    wait = self._fetch_retry_wait * (attempt + 1)
                    self.log_message(
                        f"Rate limited on {symbol}. Waiting {wait}s "
                        f"(attempt {attempt+1}/{self._fetch_retries})"
                    )
                    time.sleep(wait)
                else:
                    self.log_message(f"Error fetching {symbol}: {e}")
                    return None
        self.log_message(f"Failed to fetch {symbol} after {self._fetch_retries} attempts.")
        return None

    # -------------------------------------------------------------------------
    # Core strategy
    # -------------------------------------------------------------------------

    def _run_momentum_strategy(self, market_condition):
        """
        For each stock in the ranked universe:
        - Only enter new positions when market is Bullish
        - Always check exits regardless of regime
        - Enter if SMA20 > SMA50 and RSI > 50 and MACD > signal
        - Exit if SMA20 < SMA50 or RSI < 45
        """
        ranked = self._rank_assets()

        for stock in ranked:
            total_trades = self.day_trades_count["buy"] + self.day_trades_count["sell"]
            if total_trades >= self.max_daily_trades:
                self.log_message(f"Daily trade limit ({self.max_daily_trades}) reached.")
                break

            df = self._price_cache.get(stock)
            if df is None:
                self.log_message(f"No cached data for {stock}, skipping.")
                continue

            sma_fast = df["close"].rolling(self.sma_fast).mean().iloc[-1]
            sma_slow = df["close"].rolling(self.sma_slow).mean().iloc[-1]
            rsi      = calculate_rsi(df["close"], period=self.rsi_period).iloc[-1]
            macd, signal = calculate_macd(df["close"])
            macd_val   = macd.iloc[-1]
            signal_val = signal.iloc[-1]

            if pd.isna(sma_fast) or pd.isna(sma_slow) or pd.isna(rsi) or pd.isna(macd_val):
                self.log_message(f"Insufficient data for {stock}. Skipping.")
                continue

            last_price = self.get_last_price(stock)
            if last_price is None or last_price <= 0:
                self.log_message(f"Invalid price for {stock}. Skipping.")
                continue

            position = self.get_position(stock)
            current_quantity = position.quantity if position else 0

            self.log_message(
                f"{stock} | SMA{self.sma_fast}: {sma_fast:.2f} | SMA{self.sma_slow}: {sma_slow:.2f} "
                f"| RSI: {rsi:.1f} | MACD: {macd_val:.2f} > Sig: {signal_val:.2f} | Pos: {current_quantity}"
            )

            # Exit: always check regardless of regime
            if current_quantity > 0 and (sma_fast < sma_slow or rsi < self.rsi_exit):
                reason = "SMA crossover" if sma_fast < sma_slow else "RSI exit"
                self.log_message(f"Exit signal for {stock}: {reason}")
                self._close_position(stock)

            # Entry: only allowed when market is Bullish
            elif market_condition == MarketCondition.Bullish:
                if (sma_fast > sma_slow and rsi > self.rsi_entry
                        and macd_val > signal_val and current_quantity == 0):
                    quantity = self._calculate_quantity(last_price)
                    if quantity > 0:
                        self._place_buy(stock, quantity)

            else:
                if current_quantity == 0:
                    self.log_message(
                        f"{stock} | Regime {market_condition.value} — no new entries."
                    )

    # -------------------------------------------------------------------------
    # Order helpers
    # -------------------------------------------------------------------------

    def _place_buy(self, stock, quantity):
        """Submit a market buy order."""
        last_price = self.get_last_price(stock)
        if last_price is None or last_price <= 0:
            self.log_message(f"Invalid price for {stock}. Skipping buy.")
            return

        self.log_message(f"BUY {stock} x{quantity} @ ~{last_price:.2f}")
        try:
            order = self.create_order(
                asset=Asset(symbol=stock),
                quantity=quantity,
                side="buy",
                type="market",
            )
            self.submit_order(order)
            self._cash_committed += quantity * last_price
            self.day_trades_count["buy"] += 1
        except Exception as e:
            self.log_message(f"Error placing buy for {stock}: {e}")

    def _close_position(self, stock):
        """Submit a market sell to close the full position."""
        position = self.get_position(stock)
        if not position or position.quantity <= 0:
            self.log_message(f"No open position for {stock} to close.")
            return

        self.log_message(f"SELL {stock} x{position.quantity} (closing position)")
        try:
            order = self.create_order(
                asset=Asset(symbol=stock),
                quantity=position.quantity,
                side="sell",
                type="market",
            )
            self.submit_order(order)
            self.day_trades_count["sell"] += 1
        except Exception as e:
            self.log_message(f"Error closing position for {stock}: {e}")

    # -------------------------------------------------------------------------
    # Position sizing
    # -------------------------------------------------------------------------

    def _calculate_quantity(self, last_price):
        """
        Fixed % of portfolio per position, capped by available cash.
        Minimum 1 share if we have enough cash, to avoid skipping high-price stocks.
        Accounts for cash already committed in this iteration.
        """
        available_cash = max(0, self.get_cash() - self._cash_committed)

        if available_cash < last_price:
            return 0  # Can't afford even 1 share

        alloc = self.position_size_pct * self.get_portfolio_value()
        alloc = min(alloc, available_cash)

        quantity = int(alloc / last_price)

        # Floor at 1 share as long as we can afford it
        return max(1, quantity)

    def _rank_assets(self):
        """
        Rank universe by 20-day risk-adjusted momentum (return / volatility).
        Higher score = stronger recent momentum relative to volatility.
        """
        scores = {}
        for stock in self.universe:
            try:
                df = self._price_cache.get(stock)
                if df is None or len(df) < 21:
                    continue
                momentum   = (df["close"].iloc[-1] / df["close"].iloc[-20]) - 1
                volatility = df["close"].pct_change().rolling(20).std().iloc[-1]
                if pd.isna(volatility) or volatility == 0:
                    continue
                scores[stock] = momentum / volatility
            except Exception as e:
                self.log_message(f"Ranking error for {stock}: {e}")
        return sorted(scores, key=scores.get, reverse=True)

    # -------------------------------------------------------------------------
    # Market regime — used as entry gate
    # -------------------------------------------------------------------------

    def determine_market_condition(self):
        """
        SPY-based regime using SMA50/100 + RSI.

        Bullish:  SMA50 > SMA100 AND RSI > 50  — entries allowed
        Bearish:  SMA50 < SMA100 OR  RSI < 45  — entries blocked
        Neutral:  everything else               — entries blocked
        """
        try:
            df = self._price_cache.get("SPY")
            if df is None:
                return MarketCondition.Neutral

            df = df.copy()
            sma_50  = df["close"].rolling(50).mean().iloc[-1]
            sma_100 = df["close"].rolling(100).mean().iloc[-1]
            rsi     = calculate_rsi(df["close"], period=14).iloc[-1]

            if pd.isna(sma_50) or pd.isna(sma_100) or pd.isna(rsi):
                return MarketCondition.Neutral

            if sma_50 > sma_100 and rsi > 50:
                return MarketCondition.Bullish
            elif sma_50 < sma_100 or rsi < 45:
                return MarketCondition.Bearish
            else:
                return MarketCondition.Neutral

        except Exception as e:
            self.log_message(f"Market condition error: {e}")
            return MarketCondition.Neutral

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def calculate_drawdown(self):
        portfolio_value = self.get_portfolio_value()
        self.portfolio_peak = max(self.portfolio_peak, portfolio_value)
        if self.portfolio_peak == 0:
            return 0
        return (portfolio_value - self.portfolio_peak) / self.portfolio_peak * 100

    def _reset_daily_state(self):
        self.day_trades_count = {"buy": 0, "sell": 0}
        self._cash_committed = 0