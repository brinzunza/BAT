import pandas as pd
from typing import Dict, Any, Optional


class MeanReversionExtremeStrategy:
    """
    Mean reversion strategy (Extreme) - Closes at opposite extreme
    Enters when price touches extreme bands, exits when price touches opposite extreme band
    More aggressive, waits for full reversal
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, **kwargs):
        self.name = "Mean Reversion (Extreme)"
        self.params = {"window": window, "num_std": num_std, **kwargs}
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Band mean reversion signals - exit at opposite extreme"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")

        df = df.copy()

        # Calculate Bollinger Bands
        df['SMA'] = df['Close'].rolling(window=self.window).mean()
        df['STD'] = df['Close'].rolling(window=self.window).std()
        df['Upper Band'] = df['SMA'] + (df['STD'] * self.num_std)
        df['Lower Band'] = df['SMA'] - (df['STD'] * self.num_std)

        # Generate signals - enter at one extreme, exit at opposite extreme
        df['Buy Signal'] = df['Close'] < df['Lower Band']
        df['Sell Signal'] = df['Close'] > df['Upper Band']

        return df

    def get_signal_names(self) -> Dict[str, str]:
        return {
            'buy': 'Buy Signal',
            'sell': 'Sell Signal'
        }

    def get_indicators(self) -> list:
        """Return list of indicator columns this strategy creates"""
        return ['SMA', 'Upper Band', 'Lower Band']

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame has required columns"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']
        return all(col in df.columns for col in required_columns)