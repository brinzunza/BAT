from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.parameters = parameters or {}
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the given data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added signal columns
        """
        pass
    
    @abstractmethod
    def get_signal_names(self) -> Dict[str, str]:
        """
        Return the signal column names used by this strategy
        
        Returns:
            Dictionary with 'buy' and 'sell' keys mapping to column names
        """
        pass
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame has required columns"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']
        return all(col in df.columns for col in required_columns)