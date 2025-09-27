import pandas as pd
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands
    Based on the meanReversion.ipynb notebook
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0, **kwargs):
        super().__init__("Mean Reversion", {"window": window, "num_std": num_std, **kwargs})
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Band mean reversion signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
            
        df = df.copy()
        
        # Calculate Bollinger Bands
        df['SMA'] = df['Close'].rolling(window=self.window).mean()
        df['STD'] = df['Close'].rolling(window=self.window).std()
        df['Upper Band'] = df['SMA'] + (df['STD'] * self.num_std)
        df['Lower Band'] = df['SMA'] - (df['STD'] * self.num_std)
        
        # Generate signals
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