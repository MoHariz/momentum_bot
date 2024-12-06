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
        self.base_risk_per_trade = 0.02
        self.risk_per_trade = self.base_risk_per_trade
        self.asset_specific_sma = {"default": (10, 30)}
        self.market_condition = "Neutral"
        self.portfolio_peak = 0

    def adjust_risk_based_on_market(self):
        """
        Adjust the risk per trade based on the detected market condition.
        """
        if self.market_condition == "Bull":
            self.risk_per_trade = self.base_risk_per_trade * 10.0
        elif self.market_condition == "Bear":
            self.risk_per_trade = self.base_risk_per_trade * 0.5
        else:
            self.risk_per_trade = self.base_risk_per_trade

    def filter_universe(self):
        """
        Filter the universe to include only stocks with sufficient historical data.
        """
        valid_stocks = []
        for stock in self.universe:
            bars = self.get_historical_prices(stock, length=252)
            if bars and not bars.df.empty and len(bars.df) >= 50:
                valid_stocks.append(stock)
            else:
                self.log_message(f"Excluding {stock}: Insufficient data.")
        self.universe = valid_stocks

    def detect_market_condition(self):
        """
        Detect the current market condition based on SPY's SMA slope and trend strength.
        """
        spy_data = self.get_historical_prices("SPY", length=252)
        if not spy_data or spy_data.df.empty:
            self.log_message("SPY data unavailable. Defaulting to Neutral market condition.")
            self.market_condition = "Neutral"
            return self.market_condition

        spy_df = spy_data.df
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
            bars = self.get_historical_prices(stock, length=252)
            if bars and not bars.df.empty:
                adx = self.calculate_adx(bars.df)
                return (15, 40) if adx.iloc[-1] > 25 else (20, 50)
        elif self.market_condition == "Bear":
            return (5, 20)
        elif self.market_condition == "Flat":
            return (10, 30)
        return self.asset_specific_sma["default"]

    def rank_assets(self):
        """
        Rank assets based on their risk-adjusted momentum.
        """
        scores = {}
        for stock in self.universe:
            try:
                bars = self.get_historical_prices(stock, length=252)
                if bars and not bars.df.empty:
                    df = bars.df
                    momentum = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
                    volatility = df["close"].rolling(20).std().iloc[-1]
                    scores[stock] = momentum / volatility
            except Exception as e:
                self.log_message(f"Error ranking {stock}: {e}")
        return sorted(scores, key=scores.get, reverse=True)

    def on_trading_iteration(self):
        """
        Main trading logic with enhanced debugging.
        """
        # Detect market condition
        self.detect_market_condition()

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

        # Rank assets
        ranked_assets = self.rank_assets()
        self.log_message(f"Ranked Assets: {ranked_assets}")
        if not ranked_assets:
            self.log_message("No assets ranked. Skipping iteration.")
            return

        # Allocate positions to top-ranked assets
        top_assets = ranked_assets[:3]
        self.allocate_positions(top_assets)

    def allocate_positions(self, top_assets):
        """
        Allocate positions to top-ranked assets with debugging.
        """
        available_cash = self.get_cash()
        total_weight = sum(1 / (index + 1) for index in range(len(top_assets)))

        for index, stock in enumerate(top_assets):
            bars = self.get_historical_prices(stock, length=252)
            if not bars or bars.df.empty:
                self.log_message(f"Skipping {stock}: Insufficient data.")
                continue

            df = bars.df
            sma_short_period, sma_long_period = self.get_asset_sma_periods(stock)
            sma_short = df["close"].rolling(sma_short_period).mean().iloc[-1]
            sma_long = df["close"].rolling(sma_long_period).mean().iloc[-1]
            atr = self.calculate_atr(df).iloc[-1]
            rsi = self.calculate_rsi(df["close"]).iloc[-1]

            self.log_message(f"{stock}: SMA Short: {sma_short:.2f}, SMA Long: {sma_long:.2f}, RSI: {rsi:.2f}, ATR: {atr:.2f}")

            if pd.isna(atr) or atr <= 0:
                self.log_message(f"{stock} skipped: Invalid ATR.")
                continue

            position = self.get_position(stock)
            current_quantity = position.quantity if position else 0
            allocation = (1 / (index + 1)) / total_weight
            risk_amount = available_cash * self.risk_per_trade * allocation
            quantity = int(risk_amount / (atr * 2))

            if sma_short > sma_long and current_quantity == 0:
                self.place_trade(stock, quantity, atr)
            elif sma_short < sma_long and current_quantity > 0:
                self.close_position(stock)

    def after_market_closes(self):
        """
        Log the portfolio value and cash available after the market closes.
        """
        self.log_message(f"Portfolio Value: {self.get_portfolio_value()}")
        self.log_message(f"Cash Available: {self.get_cash()}")

    def place_trade(self, stock, quantity, atr):
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

        order = self.create_order(
            asset=Asset(symbol=stock),
            quantity=quantity,
            type="market",
            side="buy",
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )
        self.submit_order(order)

    def close_position(self, stock):
        """
        Close the position for the specified stock.
        """
        position = self.get_position(stock)
        if position and position.quantity > 0:
            order = self.create_order(
                asset=Asset(symbol=stock),
                quantity=position.quantity,
                type="market",
                side="sell",
            )
            self.submit_order(order)

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
        portfolio_value = self.get_portfolio_value()
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
