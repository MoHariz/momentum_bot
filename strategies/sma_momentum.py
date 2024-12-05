from lumibot.strategies.strategy import Strategy
from lumibot.entities.asset import Asset
import pandas as pd


class SMAMomentumRefined(Strategy):
    def initialize(self):
        self.sleeptime = "1D"  # Run once per day
        self.universe = ["VOO", "QQQ", "GLD"] # initial capital < $5000
        # self.universe = ["SPY", "QQQ", "GLD"] # initial capital $5000 - $10000
        # self.universe = ["QQQ", "XLK", "XLV", "XLE", "VOO", "SPY", "GLD"] # initial capital > $25000+

        self.log_message("Initialized refined SMAMomentumBot with diversified asset universe.")
        self.risk_per_trade = 0.02
        # self.risk_per_trade = 0.01 # for better risk control
        self.asset_specific_sma = {"default": (10, 30)}
        self.position_entry_times = {}
        self.portfolio_peak = 0

    def get_asset_sma_periods(self, stock):
        return self.asset_specific_sma.get(stock, self.asset_specific_sma["default"])

    def calculate_drawdown(self):
        portfolio_value = self.get_portfolio_value()
        self.portfolio_peak = max(self.portfolio_peak, portfolio_value)
        drawdown = (portfolio_value - self.portfolio_peak) / self.portfolio_peak * 100
        return drawdown

    def on_trading_iteration(self):
        drawdown = self.calculate_drawdown()
        self.log_message(f"Current Drawdown: {drawdown:.2f}%")
        
        if drawdown < -20:
            self.log_message("Drawdown exceeds -20%. Pausing trading.")
            return

        for stock in self.universe:
            if not self.broker.is_market_open():
                continue

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
            adx = self.calculate_adx(df)
            current_adx = adx.iloc[-1]
            atr = self.calculate_atr(df).iloc[-1]

            position = self.get_position(stock)
            current_quantity = position.quantity if position else 0

            if sma_short > sma_long and current_rsi < 70 and current_adx > 20 and current_quantity == 0:
                portfolio_value = self.get_portfolio_value()
                risk_amount = portfolio_value * self.risk_per_trade
                quantity = int(risk_amount / (atr * 2))
                self.place_trade(stock, quantity, atr)

            elif sma_short < sma_long and current_quantity > 0:
                self.log_message(f"Bearish crossover for {stock}. Closing position.")
                self.close_position(stock)

    def place_trade(self, stock, quantity, atr):
        last_price = self.get_last_price(stock)
        if last_price is None or last_price <= 0:
            self.log_message(f"Invalid last price for {stock}. Skipping trade.")
            return

        stop_loss_price = last_price - 1.5 * atr
        take_profit_price = last_price + 2 * atr

        self.log_message(f"Placing buy order for {stock} with quantity {quantity}.")
        
        # Ensure quantity is an integer and greater than 0
        quantity = int(quantity)
        if quantity <= 0:
            self.log_message(f"Quantity for {stock} is invalid. Skipping trade.")
            return

        # Create and submit the order
        try:
            order = self.create_order(
                asset=Asset(symbol=stock),
                quantity=quantity,
                type="market",
                side="buy",  # Correctly specify the side as a string
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
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
                    side="sell",  # Use "sell" to close the position
                )
                self.submit_order(order)
            except Exception as e:
                self.log_message(f"Error closing position for {stock}: {e}")
        else:
            self.log_message(f"No open position found for {stock} to close.")

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    def calculate_adx(self, df, period=14):
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()
        plus_di = (plus_dm / atr).rolling(period).mean() * 100
        minus_di = (minus_dm / atr).rolling(period).mean() * 100
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.rolling(period).mean()

    def calculate_atr(self, df, period=14):
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
