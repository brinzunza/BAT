import pandas as pd
import numpy as np

def sma(data, period):
    """Simple Moving Average"""
    return data.rolling(window=period).mean()

def ema(data, period):
    """Exponential Moving Average"""
    return data.ewm(span=period).mean()

def bollinger_bands(data, period=20, std_dev=2):
    """Bollinger Bands indicator"""
    sma_values = sma(data, period)
    rolling_std = data.rolling(window=period).std()

    upper_band = sma_values + (rolling_std * std_dev)
    lower_band = sma_values - (rolling_std * std_dev)

    return {
        'upper': upper_band,
        'middle': sma_values,
        'lower': lower_band
    }

def rsi(data, period=14):
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values

def macd(data, fast_period=12, slow_period=26, signal_period=9):
    """MACD indicator"""
    ema_fast = ema(data, fast_period)
    ema_slow = ema(data, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def detect_candlestick_patterns(open_prices, high_prices, low_prices, close_prices):
    """Detect basic candlestick patterns"""
    patterns = pd.Series(index=close_prices.index, dtype=str)

    # Calculate body and shadows
    body = abs(close_prices - open_prices)
    upper_shadow = high_prices - np.maximum(open_prices, close_prices)
    lower_shadow = np.minimum(open_prices, close_prices) - low_prices

    # Doji pattern (small body)
    avg_body = body.rolling(window=20).mean()
    doji_condition = body < (avg_body * 0.1)
    patterns[doji_condition] = 'doji'

    # Hammer pattern (small body, long lower shadow, small upper shadow)
    hammer_condition = (
        (body < (upper_shadow * 2)) &
        (lower_shadow > (body * 2)) &
        (upper_shadow < (body * 0.5))
    )
    patterns[hammer_condition] = 'hammer'

    # Shooting star (small body, long upper shadow, small lower shadow)
    shooting_star_condition = (
        (body < (lower_shadow * 2)) &
        (upper_shadow > (body * 2)) &
        (lower_shadow < (body * 0.5))
    )
    patterns[shooting_star_condition] = 'shooting_star'

    return patterns.fillna('none')