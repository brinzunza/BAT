import pandas as pd
import numpy as np
from typing import Union, Tuple, List
from scipy.signal import argrelextrema


def sma(data: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return data.rolling(window=window, min_periods=1).mean()


def ema(data: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average"""
    return data.ewm(span=window, adjust=False).mean()


def rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD - Moving Average Convergence Divergence"""
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Williams %R"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low))


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range"""
    high_low = high - low
    high_close_prev = np.abs(high - close.shift(1))
    low_close_prev = np.abs(low - close.shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    return pd.Series(true_range).rolling(window=window).mean()


def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Price Trend"""
    price_change = close.pct_change()
    vpt = (price_change * volume).cumsum()
    return vpt


def support_resistance_levels(data: pd.Series, window: int = 20, min_distance: int = 5) -> Tuple[List[float], List[float]]:
    """Identify support and resistance levels"""
    # Find local maxima and minima
    highs = argrelextrema(data.values, np.greater, order=min_distance)[0]
    lows = argrelextrema(data.values, np.less, order=min_distance)[0]

    resistance_levels = []
    support_levels = []

    # Get resistance levels (local maxima)
    for idx in highs:
        if idx < len(data):
            resistance_levels.append(data.iloc[idx])

    # Get support levels (local minima)
    for idx in lows:
        if idx < len(data):
            support_levels.append(data.iloc[idx])

    # Sort and remove duplicates
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
    support_levels = sorted(list(set(support_levels)))

    return support_levels, resistance_levels


def fibonacci_retracement_levels(high_price: float, low_price: float) -> dict:
    """Calculate Fibonacci retracement levels"""
    diff = high_price - low_price
    levels = {
        '0%': high_price,
        '23.6%': high_price - 0.236 * diff,
        '38.2%': high_price - 0.382 * diff,
        '50%': high_price - 0.5 * diff,
        '61.8%': high_price - 0.618 * diff,
        '100%': low_price
    }
    return levels


def detect_candlestick_patterns(open_prices: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Detect common candlestick patterns"""
    patterns = pd.DataFrame(index=open_prices.index)

    # Calculate body and shadow sizes
    body = np.abs(close - open_prices)
    upper_shadow = high - np.maximum(close, open_prices)
    lower_shadow = np.minimum(close, open_prices) - low

    # Doji: small body relative to range
    total_range = high - low
    patterns['doji'] = (body / total_range < 0.1) & (total_range > 0)

    # Hammer: small body at top, long lower shadow
    patterns['hammer'] = (
        (lower_shadow > 2 * body) &
        (upper_shadow < 0.5 * body) &
        (close > open_prices)
    )

    # Hanging Man: small body at top, long lower shadow, but bearish
    patterns['hanging_man'] = (
        (lower_shadow > 2 * body) &
        (upper_shadow < 0.5 * body) &
        (close < open_prices)
    )

    # Shooting Star: small body at bottom, long upper shadow
    patterns['shooting_star'] = (
        (upper_shadow > 2 * body) &
        (lower_shadow < 0.5 * body)
    )

    # Engulfing patterns
    prev_body = body.shift(1)
    prev_close = close.shift(1)
    prev_open = open_prices.shift(1)

    # Bullish Engulfing
    patterns['bullish_engulfing'] = (
        (close > open_prices) &  # Current candle is bullish
        (prev_close < prev_open) &  # Previous candle is bearish
        (open_prices < prev_close) &  # Current open < previous close
        (close > prev_open)  # Current close > previous open
    )

    # Bearish Engulfing
    patterns['bearish_engulfing'] = (
        (close < open_prices) &  # Current candle is bearish
        (prev_close > prev_open) &  # Previous candle is bullish
        (open_prices > prev_close) &  # Current open > previous close
        (close < prev_open)  # Current close < previous open
    )

    return patterns


def detect_head_and_shoulders(data: pd.Series, window: int = 20) -> pd.Series:
    """Detect Head and Shoulders pattern"""
    # This is a simplified version - a more sophisticated implementation
    # would analyze the specific peak relationships
    signals = pd.Series(False, index=data.index)

    for i in range(2 * window, len(data) - window):
        # Look for three peaks pattern
        left_peak_idx = data.iloc[i-2*window:i-window].idxmax()
        center_peak_idx = data.iloc[i-window:i+window].idxmax()
        right_peak_idx = data.iloc[i:i+window].idxmax()

        if (left_peak_idx in data.index and
            center_peak_idx in data.index and
            right_peak_idx in data.index):

            left_peak = data.loc[left_peak_idx]
            center_peak = data.loc[center_peak_idx]
            right_peak = data.loc[right_peak_idx]

            # Check if center peak is higher and side peaks are roughly equal
            if (center_peak > left_peak and
                center_peak > right_peak and
                abs(left_peak - right_peak) / max(left_peak, right_peak) < 0.05):
                signals.iloc[i] = True

    return signals


def detect_breakouts(data: pd.Series, volume: pd.Series, window: int = 20, volume_threshold: float = 1.5) -> Tuple[pd.Series, pd.Series]:
    """Detect breakout patterns with volume confirmation"""
    # Calculate support and resistance levels
    rolling_max = data.rolling(window=window).max()
    rolling_min = data.rolling(window=window).min()

    # Calculate average volume
    avg_volume = volume.rolling(window=window).mean()

    # Detect breakouts
    resistance_breakout = (data > rolling_max.shift(1)) & (volume > avg_volume * volume_threshold)
    support_breakdown = (data < rolling_min.shift(1)) & (volume > avg_volume * volume_threshold)

    return resistance_breakout, support_breakdown