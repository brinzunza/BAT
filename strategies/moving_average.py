import pandas as pd
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy


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
        df['short_mavg'] = df['Close'].rolling(window=self.short_window, min_periods=1).mean()
        df['medium_mavg'] = df['Close'].rolling(window=self.medium_window, min_periods=1).mean()
        df['long_mavg'] = df['Close'].rolling(window=self.long_window, min_periods=1).mean()
        
        # Generate signals based on crossover conditions
        df['Sell Signal'] = (
            (df['medium_mavg'] > df['long_mavg']) & 
            (df['short_mavg'] > df['medium_mavg'])
        )
        df['Buy Signal'] = (
            (df['medium_mavg'] < df['long_mavg']) & 
            (df['short_mavg'] < df['medium_mavg'])
        )
        
        return df
    
    def get_signal_names(self) -> Dict[str, str]:
        return {
            'buy': 'Buy Signal',
            'sell': 'Sell Signal'
        }
    
    def get_indicators(self) -> list:
        """Return list of indicator columns this strategy creates"""
        return ['short_mavg', 'medium_mavg', 'long_mavg']