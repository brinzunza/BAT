import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from .base_strategy import BaseStrategy


class TrendStructureStrategy(BaseStrategy):
    """
    Trend structure strategy based on swing highs/lows
    Based on the dtfx.ipynb notebook (fixed version)
    """
    
    def __init__(self, **kwargs):
        super().__init__("Trend Structure", kwargs)
        self.trend = 0
        self.valid_high = 0
        self.valid_low = 0
        self.potential_high = None
        self.potential_low = None
        self.signals_list = []
        
    def initial_direction(self, df: pd.DataFrame) -> int:
        """Determine initial trend direction"""
        initial = df.head(30)
        up = sum(initial['Close'].diff() > 0)
        down = sum(initial['Close'].diff() <= 0)
        return 2 if up >= down else 1
    
    def initial_high_low(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Get initial high and low values"""
        initial = df.head(30)
        high = initial['High'].max()
        low = initial['Low'].min()
        return high, low
    
    def analyze_structure(self, df: pd.DataFrame, index: int):
        """Analyze trend structure at given index"""
        if index == 0:
            return
            
        curr_candle = df.iloc[index]
        prev_candle = df.iloc[index - 1]
        
        if self.trend == 2:  # Uptrend
            if curr_candle['Close'] < prev_candle['Low'] and self.potential_high is None:
                self.potential_high = prev_candle['High']
                self.potential_low = curr_candle['Low']
            
            if self.potential_high is not None and curr_candle['Low'] < self.potential_low:
                self.potential_low = curr_candle['Low']
            
            if curr_candle['Close'] < self.valid_low:
                self.trend = 1
                self.valid_high = self.potential_high
                self.potential_high = None
                self.potential_low = None
                self.signals_list.append(('sell', self.valid_high, self.valid_low, index))
            
            if (self.potential_high is not None and 
                curr_candle['Close'] > self.potential_high):
                self.valid_high = self.potential_high
                self.valid_low = self.potential_low
                self.potential_low = None
                self.potential_high = None
                self.signals_list.append(('buy', self.valid_low, self.valid_high, index))
        
        elif self.trend == 1:  # Downtrend
            if curr_candle['Close'] > prev_candle['High'] and self.potential_low is None:
                self.potential_low = prev_candle['Low']
                self.potential_high = curr_candle['High']
            
            if self.potential_low is not None and curr_candle['High'] > self.potential_high:
                self.potential_high = curr_candle['High']
            
            if curr_candle['Close'] > self.valid_high:
                self.trend = 2
                self.valid_low = self.potential_low
                self.potential_high = None
                self.potential_low = None
                self.signals_list.append(('buy', self.valid_low, self.valid_high, index))
            
            if (self.potential_low is not None and 
                curr_candle['Close'] < self.potential_low):
                self.valid_low = self.potential_low
                self.valid_high = self.potential_high
                self.potential_low = None
                self.potential_high = None
                self.signals_list.append(('sell', self.valid_high, self.valid_low, index))
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend structure signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
            
        df = df.copy()
        
        # Reset state
        self.trend = self.initial_direction(df)
        self.valid_high, self.valid_low = self.initial_high_low(df)
        self.potential_high = None
        self.potential_low = None
        self.signals_list = []
        
        # Initialize signal columns
        df['Buy Signal'] = False
        df['Sell Signal'] = False
        
        # Analyze structure for each candle
        for i in range(len(df)):
            self.analyze_structure(df, i)
        
        # Apply signals to dataframe
        for signal_type, stop, target, index in self.signals_list:
            if signal_type == 'buy':
                df.iloc[index, df.columns.get_loc('Buy Signal')] = True
            elif signal_type == 'sell':
                df.iloc[index, df.columns.get_loc('Sell Signal')] = True
        
        return df
    
    def get_signal_names(self) -> Dict[str, str]:
        return {
            'buy': 'Buy Signal',
            'sell': 'Sell Signal'
        }
    
    def get_indicators(self) -> list:
        """Return list of indicator columns this strategy creates"""
        return []