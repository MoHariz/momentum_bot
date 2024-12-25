import pandas as pd

def calculate_rsi(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    """
    if len(prices) < period:
        return None

    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def calculate_atr(df, period=14):
    """
    Calculate the Average True Range (ATR).
    """
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift(1)),
        abs(df["low"] - df["close"].shift(1))
    ],
                    axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    atr.dropna(inplace=True)  # Handle NaN values
    return atr

def calculate_adx(df, period=14):
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

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculate MACD line and Signal line."""
    ema_short = prices.ewm(span=short_period).mean()
    ema_long = prices.ewm(span=long_period).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_period).mean()
    return macd, signal