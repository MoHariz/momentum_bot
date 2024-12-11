from lumibot.strategies.strategy import Strategy
from lumibot.entities.asset import Asset
import pandas as pd

class SMAMomentumBot(Strategy):
    """
    A dynamic SMA momentum bot that adjusts its risk per trade based on the detected market condition.
    """

    def initialize(self):
        """
        Initialize the strategy with the universe of assets and the initial risk per trade.
        """
        self.sleeptime = "1D"
        self.universe = ["VOO", "QQQ", "GLD", "USMV", "EEM", "IEFA", "AGG", "XLK", "XLV", "XLF"]
        self.base_risk_per_trade = 0.03
        self.risk_per_trade = self.base_risk_per_trade
        self.asset_specific_sma = {"default": (10, 30)}
        self.market_condition = "Neutral"
        self.portfolio_peak = 0
        # self.force_start_immediately = True # Start the bot immediately after deployment. For testing purpose

    # 1. Introduce Faster Indicators for Bull Markets
    def calculate_macd(self, prices, short_period=12, long_period=26, signal_period=9):
        """Calculate MACD line and Signal line."""
        ema_short = prices.ewm(span=short_period).mean()
        ema_long = prices.ewm(span=long_period).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=signal_period).mean()
        return macd, signal

    def detect_bull_market_trend(self, stock_data):
        prices = stock_data["close"]
        macd, signal = self.calculate_macd(prices)
        rsi = self.calculate_rsi(prices)

        is_macd_bullish = macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
        is_rsi_bullish = rsi.iloc[-1] > 50
        return is_macd_bullish and is_rsi_bullish
    
    def adjust_position_size_for_volatility(self, atr, risk_amount, last_price):
        """
        Adjust position size dynamically based on volatility (ATR) and stock price.
        """
        if last_price <= 0:
            self.log_message("Invalid last price detected. Skipping position sizing.")
            return 0

        atr_multiplier = 1.5 if self.market_condition == "Bull" else 2.5
        position_size = int(risk_amount / (atr * atr_multiplier * last_price))
        
        # Log details of calculation
        self.log_message(
            f"Calculating position size: Risk Amount={risk_amount}, ATR={atr}, ATR Multiplier={atr_multiplier}, "
            f"Last Price={last_price}, Position Size={position_size}"
        )
        return position_size
    
    # 2. Volatility-Based Filters
    def detect_high_volatility(self):
        spy_data = self.get_historical_prices("SPY", length=252)
        if not spy_data or spy_data.df.empty:
            return False

        atr = self.calculate_atr(spy_data.df).iloc[-1]
        volatility_threshold = spy_data.df["close"].mean() * 0.05
        return atr > volatility_threshold

    def adjust_for_volatility(self):
        """
        Adjust risk parameters dynamically based on volatility.
        """
        if self.detect_high_volatility():
            self.risk_per_trade = self.base_risk_per_trade * 0.5
            self.log_message("High volatility detected. Reducing risk.")
        else:
            self.risk_per_trade = self.base_risk_per_trade
    
    def adjust_risk_based_on_market(self):
        """Adjust the risk per trade based on the detected market condition."""
        risk_multipliers = {"Bull": 5.0, "Bear": 0.5, "Flat": 1.0}
        self.risk_per_trade = self.base_risk_per_trade * risk_multipliers.get(self.market_condition, 1.0)

    def get_valid_data(self, stock, length=252):
        """Fetch and validate historical price data."""
        bars = self.get_historical_prices(stock, length=length)
        if not bars or bars.df.empty or len(bars.df) < 50:
            return self.log_message(f"Skipping {stock}: Insufficient data.")
        return bars.df

    def filter_universe(self):
        """
        Filter the universe to include only stocks with sufficient historical data.
        """
        valid_stocks = []
        for stock in self.universe:
            bars = self.get_valid_data(stock, length=252)
            if bars is not None:
                valid_stocks.append(stock)
            else:
                self.log_message(f"Excluding {stock}: Insufficient data.")
        self.universe = valid_stocks

    def detect_market_condition(self):
        """
        Detect the current market condition based on SPY's SMA slope and trend strength.
        """
        spy_data = self.get_valid_data("SPY", length=252)
        if spy_data is None:
            self.log_message("SPY data unavailable. Defaulting to Neutral market condition.")
            self.market_condition = "Neutral"
            return self.market_condition

        spy_df = spy_data
        spy_sma_50 = spy_df["close"].rolling(50).mean()
        spy_sma_200 = spy_df["close"].rolling(200).mean()

        # Weighted slope for SMA 50
        recent_slopes = [spy_sma_50.iloc[-i] - spy_sma_50.iloc[-(i + 1)] for i in range(1, 6)]
        weighted_slope = sum(recent_slopes) / len(recent_slopes)

        # Dynamic thresholds
        volatility = spy_df["close"].rolling(20).std().iloc[-1]
        slope_threshold = 0.05 if volatility > 1.5 else 0.1

        # Determine market condition
        if weighted_slope > slope_threshold and spy_sma_50.iloc[-1] > spy_sma_200.iloc[-1]:
            self.market_condition = "Bull"
        elif weighted_slope < -slope_threshold and spy_sma_50.iloc[-1] < spy_sma_200.iloc[-1]:
            self.market_condition = "Bear"
        else:
            self.market_condition = "Flat"

        self.adjust_risk_based_on_market()
        self.log_message(f"Market Condition: {self.market_condition}")
        return self.market_condition

    def get_asset_sma_periods(self, stock):
        """
        Dynamically adjust SMA periods based on market condition and trend strength.
        """
        if self.market_condition == "Bull":
            bars = self.get_valid_data(stock, length=252)
            if bars is not None:
                adx = self.calculate_adx(bars)
                return (15, 40) if adx.iloc[-1] > 25 else (20, 50)
        elif self.market_condition == "Bear":
            return (5, 20)
        elif self.market_condition == "Flat":
            return (30, 70)
        return self.asset_specific_sma["default"]

    def rank_assets(self):
        """
        Rank assets based on their risk-adjusted momentum.
        """
        scores = {}
        for stock in self.universe:
            try:
                bars = self.get_valid_data(stock, length=252)
                if bars is not None:
                    df = bars
                    momentum = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
                    volatility = df["close"].rolling(20).std().iloc[-1]
                    scores[stock] = momentum / volatility
            except Exception as e:
                self.log_message(f"Error ranking {stock}: {e}")
        return sorted(scores, key=scores.get, reverse=True)

    # Core Trading Logic
    def on_trading_iteration(self):
        """
        Main trading logic.
        """
        # Adjust risk based on market conditions
        self.detect_market_condition()
        self.adjust_for_volatility()

        # Filter universe
        self.filter_universe()
        self.log_message(f"Filtered Universe: {self.universe}")
        if not self.universe:
            self.log_message("No valid stocks in universe. Skipping iteration.")
            return

        # Check drawdown
        drawdown = self.calculate_drawdown()
        self.log_message(f"Drawdown: {drawdown:.2f}%")
        if drawdown < -20:
            self.log_message("Drawdown exceeds -20%. Pausing trading.")
            return

        # Rank and trade assets
        ranked_assets = self.rank_assets()
        self.log_message(f"Ranked Assets: {ranked_assets}")
        if not ranked_assets:
            self.log_message("No assets ranked. Skipping iteration.")
            return

        # Allocate positions to top-ranked assets
        top_assets = ranked_assets[:3]
        self.allocate_positions(top_assets)

    def on_abrupt_closing(self):
        self.log_message("Abrupt closing")
        self.sell_all()

    def log_position(self, stock, allocation, risk_per_trade, available_cash, atr, total_weight, last_price, quantity):
        """
        Calculate the quantity to be traded based on the allocation and risk per trade.
        """
        self.log_message(f"Calculating quantity for {stock}: Allocation={allocation}, Risk Per Trade={risk_per_trade}, Available Cash={available_cash}, ATR={atr}, Total Weight={total_weight}, Last Price={last_price}, Calculated Quantity={quantity}")

    def allocate_positions(self, top_assets):
        """
        Allocate positions to top-ranked assets, with enhanced bull market logic.
        """
        available_cash = self.cash
        total_weight = sum(1 / (index + 1) for index in range(len(top_assets)))

        for index, stock in enumerate(top_assets):
            bars = self.get_historical_prices(stock, length=252)
            if not bars or bars.df.empty:
                self.log_message(f"Skipping {stock}: Insufficient data.")
                continue

            df = bars.df
            sma_short, sma_long = self.get_asset_sma_periods(stock)
            sma_short_val = df["close"].rolling(sma_short).mean().iloc[-1]
            sma_long_val = df["close"].rolling(sma_long).mean().iloc[-1]
            atr = self.calculate_atr(df).iloc[-1]
            last_price = self.get_last_price(stock)  # Fetch the last price of the stock

            if pd.isna(atr) or atr <= 0:
                self.log_message(f"{stock} skipped: Invalid ATR ({atr}).")
                continue
            if last_price <= 0:
                self.log_message(f"{stock} skipped: Invalid Last Price ({last_price}).")
                continue

            allocation = 1 / (index + 1)**0.5 / total_weight
            risk_amount = available_cash * self.risk_per_trade * allocation

            # Adjust position size using updated method
            quantity = self.adjust_position_size_for_volatility(atr, risk_amount, last_price)

            position = self.get_position(stock)
            current_quantity = position.quantity if position else 0
            self.log_position(stock, allocation, self.risk_per_trade, available_cash, atr, total_weight, last_price, quantity)

            if current_quantity > 0:
                if sma_short_val > sma_long_val:
                    self.place_trade(stock, quantity, atr, last_price)
                elif sma_short_val < sma_long_val:
                    self.close_position(stock, last_price)
            else:
                self.log_message(f"Quantity for {stock}: {quantity}. Skipping trade.")

    def get_account_value(self):
        """
        Get the current account value and cash available.
        """
        return self.portfolio_value, self.cash

    def before_market_opens(self):
        """
        Log the portfolio value and cash available before the market opens.
        """
        value, cash = self.get_account_value()
        self.detect_market_condition()
        self.log_message("Before Market Opens")
        self.log_message(f"Portfolio Value: {value}")
        self.log_message(f"Cash Available: {cash}")
        self.log_message(f"Potential Market Condition: {self.market_condition}")

    def after_market_closes(self):
        """
        Log the portfolio value and cash available after the market closes.
        """
        value, cash = self.get_account_value()
        self.log_message("After Market Closes")
        self.log_message(f"Portfolio Value: {value}")
        self.log_message(f"Cash Available: {cash}")

    def place_trade(self, stock, quantity, atr, last_price):
        """
        Place a trade with the specified quantity and risk management parameters.
        """
        if quantity <= 0:
            return

        last_price = self.get_last_price(stock)
        if last_price <= 0:
            return

        stop_loss_price = last_price - 1.5 * atr
        take_profit_price = last_price + 2 * atr

        self.log_message(f"Placing trade for {stock} (${last_price}/per share) with current cash at {self.cash}: Quantity={quantity}, ATR={atr}, Last Price={last_price}")
        order = self.create_order(
            asset=Asset(symbol=stock),
            quantity=quantity,
            type="market",
            side="buy",
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )
        try:
            self.submit_order(order)
        except Exception as e:
            self.log_message(f"Error placing trade for {stock}: {e}")

    def close_position(self, stock, last_price):
        """
        Close the position for the specified stock.
        """
        position = self.get_position(stock)
        if position and position.quantity > 0:
            self.log_message(f"Closing position for {stock} (${last_price}/per share): Quantity={position.quantity}")
            order = self.create_order(
                asset=Asset(symbol=stock),
                quantity=position.quantity,
                type="market",
                side="sell",
            )
            try:
                self.submit_order(order)
            except Exception as e:
                self.log_message(f"Error closing position for {stock}: {e}")

    def calculate_rsi(self, prices, period=14):
        """
        Calculate the Relative Strength Index (RSI).
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    def calculate_atr(self, df, period=14):
        """
        Calculate the Average True Range (ATR).
        """
        tr = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_drawdown(self):
        """
        Calculate the current drawdown.
        """
        portfolio_value = self.portfolio_value
        self.portfolio_peak = max(self.portfolio_peak, portfolio_value)
        return (portfolio_value - self.portfolio_peak) / self.portfolio_peak * 100

    def calculate_adx(self, df, period=14):
        """
        Calculate the Average Directional Index (ADX).
        """
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.rolling(window=period).mean()

    def calculate_rsi(self, prices, period=14):
        """
        Calculate the Relative Strength Index (RSI).
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

