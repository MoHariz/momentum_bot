import pandas as pd


def calculate_rsi(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    Returns a Series of RSI values, or a Series of NaN if insufficient data.
    """
    if len(prices) < period:
        return pd.Series([float("nan")] * len(prices), index=prices.index)

    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def calculate_atr(df, period=14):
    """
    Calculate the Average True Range (ATR).
    Returns a full-length Series aligned to df's index (NaN for early rows).
    Note: does NOT drop NaN rows to preserve index alignment with the source df.
    """
    tr = pd.concat(
        [
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window=period).mean()


def calculate_adx(df, period=14):
    """
    Calculate the Average Directional Index (ADX).
    Returns a Series aligned to df's index.
    """
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
    ).max(axis=1)
    atr = tr.rolling(window=period).mean()

    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Avoid division by zero
    with pd.option_context("mode.use_inf_as_na", True):
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

    return dx.rolling(window=period).mean()


def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """
    Calculate MACD line and Signal line.
    Returns (macd, signal) as a tuple of Series.
    """
    ema_short = prices.ewm(span=short_period, adjust=False).mean()
    ema_long = prices.ewm(span=long_period, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal


def calculate_sma_slope(series, period, threshold=0.1):
    """
    Calculate whether the SMA over `period` bars is sloping up, down, or flat.
    Returns 1 (up), -1 (down), or 0 (flat) based on the threshold.
    """
    sma = series.rolling(period).mean()
    if len(sma.dropna()) < 2:
        return 0
    slope = (sma.iloc[-1] - sma.iloc[-2]) / sma.iloc[-2] * 100
    if slope > threshold:
        return 1
    elif slope < -threshold:
        return -1
    return 0