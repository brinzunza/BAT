import pandas as pd
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
from indicators.technical_indicators import detect_candlestick_patterns


class CandlestickPatternsStrategy(BaseStrategy):
    """
    Candlestick Patterns strategy
    Buy on bullish patterns (hammer, bullish engulfing)
    Sell on bearish patterns (shooting star, bearish engulfing, hanging man)
    """

    def __init__(self, **kwargs):
        super().__init__("Candlestick Patterns", {**kwargs})

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate candlestick pattern trading signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")

        df = df.copy()

        # Detect candlestick patterns
        patterns = detect_candlestick_patterns(df['Open'], df['High'], df['Low'], df['Close'])

        # Add pattern columns to main dataframe
        for pattern in patterns.columns:
            df[pattern] = patterns[pattern]

        # Generate signals based on patterns
        # Bullish patterns
        df['Buy Signal'] = (
            df['hammer'] |
            df['bullish_engulfing'] |
            df['doji']  # Doji can signal reversal, treat as neutral-bullish in context
        )

        # Bearish patterns
        df['Sell Signal'] = (
            df['shooting_star'] |
            df['hanging_man'] |
            df['bearish_engulfing']
        )

        return df

    def get_signal_names(self) -> Dict[str, str]:
        return {
            'buy': 'Buy Signal',
            'sell': 'Sell Signal'
        }

    def get_indicators(self) -> list:
        """Return list of indicator columns this strategy creates"""
        return ['doji', 'hammer', 'hanging_man', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing']