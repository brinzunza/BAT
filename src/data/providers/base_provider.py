from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
from datetime import datetime


class BaseDataProvider(ABC):
    """Base class for all data providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @abstractmethod
    def get_data(self, 
                 ticker: str,
                 timespan: str = 'minute',
                 from_date: str = None,
                 to_date: str = None,
                 limit: int = 50000) -> pd.DataFrame:
        """
        Get historical data for a ticker
        
        Args:
            ticker: The ticker symbol
            timespan: Time interval (minute, hour, day, etc.)
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            limit: Maximum number of records
            
        Returns:
            DataFrame with OHLCV data and timestamp
        """
        pass
    
    @abstractmethod
    def get_live_data(self, ticker: str) -> pd.DataFrame:
        """
        Get live/current data for a ticker
        
        Args:
            ticker: The ticker symbol
            
        Returns:
            DataFrame with current OHLCV data
        """
        pass
    
    def validate_response(self, data: dict) -> bool:
        """Validate API response"""
        return 'results' in data and data['results'] is not None
    
    def format_dataframe(self, data: dict) -> pd.DataFrame:
        """Convert API response to standardized DataFrame format"""
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        df.drop(columns=['vw', 'n', 't'], inplace=True, errors='ignore')
        return df