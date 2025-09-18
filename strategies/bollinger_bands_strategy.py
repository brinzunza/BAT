import pandas as pd
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
from indicators.technical_indicators import bollinger_bands


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands strategy
    Buy when price touches lower band (oversold)
    Sell when price touches upper band (overbought)
    """

    def __init__(self, window: int = 20, num_std: float = 2, **kwargs):
        super().__init__("Bollinger Bands", {
            "window": window,
            "num_std": num_std,
            **kwargs
        })
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Bands trading signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")

        df = df.copy()

        # Calculate Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(df['Close'], self.window, self.num_std)

        # Generate signals
        # Buy when price touches or goes below lower band
        df['Buy Signal'] = df['Close'] <= df['bb_lower']

        # Sell when price touches or goes above upper band
        df['Sell Signal'] = df['Close'] >= df['bb_upper']

        return df

    def get_signal_names(self) -> Dict[str, str]:
        return {
            'buy': 'Buy Signal',
            'sell': 'Sell Signal'
        }

    def get_indicators(self) -> list:
        """Return list of indicator columns this strategy creates"""
        return ['bb_upper', 'bb_middle', 'bb_lower']