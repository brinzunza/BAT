import pandas as pd
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
from indicators.technical_indicators import sma


class MovingAverageStrategy(BaseStrategy):
    """
    Moving average crossover strategy
    Based on the movingAverage.ipynb notebook
    """
    
    def __init__(self, short_window: int = 1, medium_window: int = 5, long_window: int = 25, **kwargs):
        super().__init__("Moving Average", {
            "short_window": short_window,
            "medium_window": medium_window, 
            "long_window": long_window,
            **kwargs
        })
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate moving average crossover signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")

        df = df.copy()

        # Calculate moving averages
        df['short_mavg'] = sma(df['Close'], self.short_window)
        df['medium_mavg'] = sma(df['Close'], self.medium_window)
        df['long_mavg'] = sma(df['Close'], self.long_window)

        # Detect crossover events and alignment changes
        # Current bullish alignment: short > medium > long
        bullish_alignment = (df['short_mavg'] > df['medium_mavg']) & (df['medium_mavg'] > df['long_mavg'])
        # Previous bullish alignment
        prev_bullish_alignment = (df['short_mavg'].shift(1) > df['medium_mavg'].shift(1)) & (df['medium_mavg'].shift(1) > df['long_mavg'].shift(1))

        # Current bearish alignment: short < medium < long
        bearish_alignment = (df['short_mavg'] < df['medium_mavg']) & (df['medium_mavg'] < df['long_mavg'])
        # Previous bearish alignment
        prev_bearish_alignment = (df['short_mavg'].shift(1) < df['medium_mavg'].shift(1)) & (df['medium_mavg'].shift(1) < df['long_mavg'].shift(1))

        # Buy signal: Transition to bullish alignment (all MAs cross upwards)
        df['Buy Signal'] = bullish_alignment & ~prev_bullish_alignment

        # Sell signal: Transition to bearish alignment (all MAs cross downwards)
        df['Sell Signal'] = bearish_alignment & ~prev_bearish_alignment

        return df
    
    def get_signal_names(self) -> Dict[str, str]:
        return {
            'buy': 'Buy Signal',
            'sell': 'Sell Signal'
        }
    
    def get_indicators(self) -> list:
        """Return list of indicator columns this strategy creates"""
        return ['short_mavg', 'medium_mavg', 'long_mavg']