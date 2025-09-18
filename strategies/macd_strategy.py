import pandas as pd
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
from indicators.technical_indicators import macd


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) strategy
    Buy when MACD line crosses above signal line
    Sell when MACD line crosses below signal line
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs):
        super().__init__("MACD", {
            "fast": fast,
            "slow": slow,
            "signal": signal,
            **kwargs
        })
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD-based trading signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")

        df = df.copy()

        # Calculate MACD
        df['macd_line'], df['signal_line'], df['histogram'] = macd(df['Close'], self.fast, self.slow, self.signal)

        # Generate crossover signals
        df['macd_cross_above'] = (df['macd_line'] > df['signal_line']) & (df['macd_line'].shift(1) <= df['signal_line'].shift(1))
        df['macd_cross_below'] = (df['macd_line'] < df['signal_line']) & (df['macd_line'].shift(1) >= df['signal_line'].shift(1))

        # Generate signals
        df['Buy Signal'] = df['macd_cross_above']
        df['Sell Signal'] = df['macd_cross_below']

        return df

    def get_signal_names(self) -> Dict[str, str]:
        return {
            'buy': 'Buy Signal',
            'sell': 'Sell Signal'
        }

    def get_indicators(self) -> list:
        """Return list of indicator columns this strategy creates"""
        return ['macd_line', 'signal_line', 'histogram']