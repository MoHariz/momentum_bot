from lumibot.strategies.strategy import Strategy
from lumibot.entities.asset import Asset
import pandas as pd

class SMAMomentumBot(Strategy):
    def initialize(self):
        self.sleeptime = "1D"  # Run once per day
        self.universe = ["VOO", "QQQ", "GLD", "USMV", "EEM", "IEFA", "AGG", "XLK", "XLV", "XLF"]
        self.log_message("Initialized Dynamic SMA Momentum Bot with dynamic SMA periods.")
        self.base_risk_per_trade = 0.02  # Base risk percentage for allocation
        self.risk_per_trade = self.base_risk_per_trade  # Initial risk
        self.asset_specific_sma = {"default": (10, 30)}
        self.market_condition = "Neutral"  # Default market condition
        self.portfolio_peak = 0

    def adjust_risk_based_on_market(self):
        """Adjust risk allocation dynamically based on the detected market condition."""
        if self.market_condition == "Bull":
            self.risk_per_trade = self.base_risk_per_trade * 5.0  # Aggressive risk during bull markets
        elif self.market_condition == "Bear":
            self.risk_per_trade = self.base_risk_per_trade * 0.5  # Conservative risk during bear markets
        else:
            self.risk_per_trade = self.base_risk_per_trade

        self.log_message(f"Adjusted Risk per Trade: {self.risk_per_trade:.2%} for Market Condition: {self.market_condition}")

    def detect_market_condition(self):
        """Detects the current market condition based on SPY's SMA slope."""
        spy_data = self.get_historical_prices("SPY", length=60)
        if spy_data is None or spy_data.df.empty:
            self.log_message("Error fetching SPY data for market condition detection.")
            return "Neutral"
        
        spy_df = spy_data.df
        spy_sma_50 = spy_df["close"].rolling(50).mean()
        spy_slope = spy_sma_50.iloc[-1] - spy_sma_50.iloc[-2]

        # Determine market condition based on SMA slope
        if spy_slope > 0.1:  # Positive slope
            self.market_condition = "Bull"
        elif spy_slope < -0.1:  # Negative slope
            self.market_condition = "Bear"
        else:
            self.market_condition = "Flat"

        # Adjust risk based on the detected market condition
        self.adjust_risk_based_on_market()
        self.log_message(f"Detected Market Condition: {self.market_condition}")
        return self.market_condition

    def get_asset_sma_periods(self, stock):
        """Dynamically adjusts SMA periods based on market condition and trend strength."""
        if self.market_condition == "Bull":
            bars = self.get_historical_prices(stock, length=252)
            df = bars.df
            adx = self.calculate_adx(df)
            if adx.iloc[-1] > 25:
                return (15, 40)  # More responsive SMA for strong bull trends
            else:
                return (20, 50)  # Moderate trend
        elif self.market_condition == "Bear":
            return (5, 20)  # Quick reaction for bear markets
        elif self.market_condition == "Flat":
            return (10, 30)  # Balanced SMA for flat markets
        else:
            return self.asset_specific_sma["default"]
        
    def rank_assets(self):
        """Rank assets based on risk-adjusted momentum."""
        scores = {}
        for stock in self.universe:
            try:
                bars = self.get_historical_prices(stock, length=252)  # 1 year
                df = bars.df
                momentum = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
                volatility = df["close"].rolling(20).std().iloc[-1]
                scores[stock] = momentum / volatility  # Risk-adjusted return
            except Exception as e:
                self.log_message(f"Error fetching data for {stock}: {e}")
                continue
        return sorted(scores, key=scores.get, reverse=True)
    
    def on_trading_iteration(self):
        """Main trading logic."""
        self.detect_market_condition()  # Update market condition
        drawdown = self.calculate_drawdown()
        self.log_message(f"Current Drawdown: {drawdown:.2f}%")

        if drawdown < -20:
            self.log_message("Drawdown exceeds -20%. Pausing trading.")
            return

        # Rank assets by risk-adjusted return
        ranked_assets = self.rank_assets()
        top_assets = ranked_assets[:3]

        # Total cash available
        available_cash = self.get_cash()
        total_weight = sum([1 / (index + 1) for index in range(len(top_assets))])

        for index, stock in enumerate(top_assets):
            try:
                bars = self.get_historical_prices(stock, length=200)
                df = bars.df
            except Exception as e:
                self.log_message(f"Error fetching data for {stock}: {e}")
                continue

            sma_short_period, sma_long_period = self.get_asset_sma_periods(stock)
            sma_short = df["close"].rolling(sma_short_period).mean().iloc[-1]
            sma_long = df["close"].rolling(sma_long_period).mean().iloc[-1]
            rsi = self.calculate_rsi(df["close"])
            current_rsi = rsi.iloc[-1]
            atr = self.calculate_atr(df).iloc[-1]

            # Ensure ATR is valid
            if pd.isna(atr) or atr <= 0:
                self.log_message(f"ATR calculation for {stock} is invalid. Skipping.")
                continue

            position = self.get_position(stock)
            current_quantity = position.quantity if position else 0

            allocation = (1 / (index + 1)) / total_weight
            risk_amount = available_cash * self.risk_per_trade * allocation
            quantity = int(risk_amount / (atr * 2))

            if sma_short > sma_long and current_rsi < 70 and current_quantity == 0:
                self.place_trade(stock, quantity, atr)
            elif sma_short < sma_long and current_quantity > 0:
                self.log_message(f"Bearish crossover for {stock}. Closing position.")
                self.close_position(stock)

    def after_market_closes(self):
        self.log_message("The market is closed.")
        self.log_message(f"Portfolio Value: {self.get_portfolio_value()}")
        self.log_message(f"Cash Available: {self.get_cash()}")

        # Log market conditions
        market_condition = self.detect_market_condition()
        self.log_message(f"Market Condition: {market_condition}")

    def place_trade(self, stock, quantity, atr):
        if quantity <= 0:
            self.log_message(f"Quantity for {stock} is zero or invalid. Skipping trade.")
            return

        last_price = self.get_last_price(stock)
        if last_price is None or last_price <= 0:
            self.log_message(f"Invalid last price for {stock}. Skipping trade.")
            return

        stop_loss_price = last_price - 1.5 * atr
        take_profit_price = last_price + 2 * atr

        self.log_message(f"Placing buy order for {stock} with quantity {quantity}.")
        try:
            order = self.create_order(
                asset=Asset(symbol=stock),
                quantity=quantity,
                type="market",
                side="buy",
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )
            self.submit_order(order)
        except Exception as e:
            self.log_message(f"Error placing order for {stock}: {e}")

    def close_position(self, stock):
        position = self.get_position(stock)
        if position and position.quantity > 0:
            self.log_message(f"Closing position for {stock}, Quantity: {position.quantity}")
            try:
                order = self.create_order(
                    asset=Asset(symbol=stock),
                    quantity=position.quantity,
                    type="market",
                    side="sell",
                )
                self.submit_order(order)
            except Exception as e:
                self.log_message(f"Error closing position for {stock}: {e}")

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    def calculate_atr(self, df, period=14):
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_drawdown(self):
        portfolio_value = self.get_portfolio_value()
        self.portfolio_peak = max(self.portfolio_peak, portfolio_value)
        drawdown = (portfolio_value - self.portfolio_peak) / self.portfolio_peak * 100
        return drawdown
    
    def calculate_adx(self, df, period=14):
        """
        Calculate the Average Directional Index (ADX).
        :param df: DataFrame with 'high', 'low', and 'close' columns.
        :param period: Lookback period for ADX calculation.
        :return: Series containing the ADX values.
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate True Range (TR)
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        # Calculate Directional Movement (DM)
        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()

        # Apply Wilder's smoothing for TR, +DM, and -DM
        atr = tr.rolling(window=period).mean()  # Average True Range
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()

        # Calculate Directional Indicators (+DI and -DI)
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx
    