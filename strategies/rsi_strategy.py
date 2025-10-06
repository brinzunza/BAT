import pandas as pd
from typing import Dict, Any, Optional
from indicators.technical_indicators import rsi


class RSIStrategy:
    """
    RSI (Relative Strength Index) strategy
    Buy when RSI is oversold (< oversold_threshold)
    Sell when RSI is overbought (> overbought_threshold)
    """

    def __init__(self, window: int = 14, oversold_threshold: float = 30, overbought_threshold: float = 70, **kwargs):
        self.name = "RSI"
        self.params = {
            "window": window,
            "oversold_threshold": oversold_threshold,
            "overbought_threshold": overbought_threshold,
            **kwargs
        }
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

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame has required columns"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']
        return all(col in df.columns for col in required_columns)