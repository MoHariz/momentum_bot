from lumibot.strategies.strategy import Strategy
from lumibot.entities.asset import Asset
import pandas as pd

class SimpleMomentumBot(Strategy):
    def initialize(self):
        self.sleeptime = "1D"  # Run once per day
        self.universe = ["NVDA", "AAPL", "MSFT"]

        self.asset_specific_sma = {"default": (10, 30)}
        self.portfolio_peak = 0
        self.stop_loss_multiplier = 1.5
        self.take_profit_multiplier = 2
        self.risk_per_trade = 0.02

    def get_asset_sma_periods(self, stock):
        return self.asset_specific_sma.get(stock, self.asset_specific_sma["default"])

    def calculate_drawdown(self):
        portfolio_value = self.get_portfolio_value()
        self.portfolio_peak = max(self.portfolio_peak, portfolio_value)
        drawdown = (portfolio_value - self.portfolio_peak) / self.portfolio_peak * 100
        return drawdown

    def rank_assets(self):
        """
        Rank assets based on their risk-adjusted momentum.
        """
        scores = {}
        for stock in self.universe:
            try:
                bars = self.get_historical_prices(stock, length=252)  # 1 year
                df = bars.df
                momentum = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
                volatility = df["close"].rolling(20).std().iloc[-1]  # Last 20 days
                scores[stock] = momentum / volatility
            except Exception as e:
                self.log_message(f"Error fetching data for {stock}: {e}")
                continue
        return sorted(scores, key=scores.get, reverse=True)
    
    def before_market_opens(self):
        self.log_message("Before Market Opens")
        self.log_message(f"The total value of our portfolio is {self.get_portfolio_value()}")
        self.log_message(f"The amount of cash we have is {self.get_cash()}")

    def on_trading_iteration(self):
        """
        Main trading logic.
        """
        # Calculate current drawdown
        drawdown = self.calculate_drawdown()
        self.log_message(f"Current Drawdown: {drawdown:.2f}%")
        
        # Stop trading if drawdown exceeds the threshold
        if drawdown < -20:
            self.log_message("Drawdown exceeds -20%. Pausing trading.")
            return

        # Rank assets by risk-adjusted return
        ranked_assets = self.rank_assets()
        top_assets = ranked_assets[:3]  # Limit to top 3 assets

        for index, stock in enumerate(top_assets):
            try:
                # Fetch historical data
                bars = self.get_historical_prices(stock, length=252)
                df = bars.df
            except Exception as e:
                self.log_message(f"Error fetching data for {stock}: {e}")
                continue

            # Calculate SMAs and ATR
            sma_short_period, sma_long_period = self.get_asset_sma_periods(stock)
            sma_short = df["close"].rolling(sma_short_period).mean().iloc[-1]
            sma_long = df["close"].rolling(sma_long_period).mean().iloc[-1]
            atr = self.calculate_atr(df).iloc[-1]

            # Get the last price and current position
            last_price = self.get_last_price(stock)
            position = self.get_position(stock)
            current_quantity = position.quantity if position else 0

            # Safeguard checks for valid data
            if atr <= 0 or last_price <= 0:
                self.log_message(f"Skipping trade for {stock}: Invalid data. ATR={atr}, Last Price={last_price}")
                continue

            # Calculate maximum risk per trade as 2% of portfolio value
            portfolio_risk = self.risk_per_trade * self.get_portfolio_value()
            max_risk = min(portfolio_risk, self.get_cash())  # Use the smaller of portfolio risk or available cash

            # Calculate position size
            risk_per_share = atr * self.stop_loss_multiplier
            if risk_per_share > 0:
                quantity = int(max_risk / risk_per_share)
                max_quantity_by_cash = int(self.get_cash() / last_price)
                quantity = min(quantity, max_quantity_by_cash)  # Ensure quantity fits within available cash
            else:
                self.log_message(f"Skipping trade for {stock}: Risk per share is zero or negative. Risk per Share={risk_per_share}")
                continue

            # Log calculation details for debugging
            self.log_message(f"Calculated quantity for {stock}: {quantity}, Max Risk: {max_risk}, Risk per Share: {risk_per_share}, Last Price: {last_price}")

            # Trading logic
            if sma_short > sma_long and current_quantity == 0:
                self.place_trade(stock, quantity)
            elif sma_short < sma_long and current_quantity > 0:
                self.close_position(stock)

    def after_market_closes(self):
        self.log_message("The market is closed")
        self.log_message(f"The total value of our portfolio is {self.get_portfolio_value()}")
        self.log_message(f"The amount of cash we have is {self.get_cash()}")

    def place_trade(self, stock, quantity):
        last_price = self.get_last_price(stock)

        if last_price is None or last_price <= 0:
            self.log_message(f"Invalid last price for {stock}. Skipping trade.")
            return

        self.log_message(f"Placing buy order for {stock} with quantity {quantity}, Last Price: {last_price}.")
        
        # Ensure quantity is an integer and greater than 0
        if quantity <= 0:
            self.log_message(f"Quantity for {stock} is invalid. Skipping trade.")
            return

        # Create and submit the order
        try:
            order = self.create_order(
                asset=Asset(symbol=stock),
                quantity=quantity,
                type="market",
                side="buy",  # Correctly specify the side as a string,
                trail_percent=5
            )
            self.submit_order(order)
        except Exception as e:
            self.log_message(f"Error placing order for {stock}: {e}")

    def close_position(self, stock):
        """Closes an open position for a specific stock."""
        position = self.get_position(stock)
        
        if position and position.quantity > 0:
            self.log_message(f"Closing position for {stock}, Quantity: {position.quantity}")
            try:
                order = self.create_order(
                    asset=Asset(symbol=stock),
                    quantity=position.quantity,
                    type="market",
                    side="sell",  # Use "sell" to close the position,
                    trail_percent=5
                )
                self.submit_order(order)
            except Exception as e:
                self.log_message(f"Error closing position for {stock}: {e}")
        else:
            self.log_message(f"No open position found for {stock} to close.")

    def calculate_atr(self, df, period=14):
        """
        Calculate the Average True Range (ATR).
        """
        tr = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        atr.dropna(inplace=True)  # Handle NaN values
        return atr
