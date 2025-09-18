import pandas as pd
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
from indicators.technical_indicators import rsi


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) strategy
    Buy when RSI is oversold (< oversold_threshold)
    Sell when RSI is overbought (> overbought_threshold)
    """

    def __init__(self, window: int = 14, oversold_threshold: float = 30, overbought_threshold: float = 70, **kwargs):
        super().__init__("RSI", {
            "window": window,
            "oversold_threshold": oversold_threshold,
            "overbought_threshold": overbought_threshold,
            **kwargs
        })
        self.window = window
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-based trading signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")

        df = df.copy()

        # Calculate RSI
        df['rsi'] = rsi(df['Close'], self.window)

        # Generate signals
        df['Buy Signal'] = df['rsi'] < self.oversold_threshold
        df['Sell Signal'] = df['rsi'] > self.overbought_threshold

        return df

    def get_signal_names(self) -> Dict[str, str]:
        return {
            'buy': 'Buy Signal',
            'sell': 'Sell Signal'
        }

    def get_indicators(self) -> list:
        """Return list of indicator columns this strategy creates"""
        return ['rsi']