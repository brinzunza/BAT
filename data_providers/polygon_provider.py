import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional
from .base_provider import BaseDataProvider


class PolygonDataProvider(BaseDataProvider):
    """Polygon.io data provider"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
    
    def get_data(self, 
                 ticker: str = 'C:EURUSD',
                 timespan: str = 'minute',
                 from_date: Optional[str] = None,
                 to_date: Optional[str] = None,
                 limit: int = 50000) -> pd.DataFrame:
        """Get historical data from Polygon API"""
        
        # Use default dates if not provided
        if not to_date:
            to_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        url = (f"{self.base_url}/{ticker}/range/1/{timespan}/{from_date}/{to_date}"
               f"?adjusted=true&sort=asc&limit={limit}&apiKey={self.api_key}")
        
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        data = response.json()
        
        if not self.validate_response(data):
            raise Exception("Invalid API response format")
        
        return self.format_dataframe(data)
    
    def get_live_data(self, ticker: str = 'C:EURUSD') -> pd.DataFrame:
        """Get current day data (simulates live data)"""
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        return self.get_data(ticker, 'minute', yesterday, today, 1440)  # 24 hours of minute data
    
    def get_crypto_data(self, 
                       ticker: str = 'X:BTCUSD',
                       timespan: str = 'minute',
                       from_date: Optional[str] = None,
                       to_date: Optional[str] = None,
                       limit: int = 50000) -> pd.DataFrame:
        """Specialized method for crypto data"""
        return self.get_data(ticker, timespan, from_date, to_date, limit)
    
    def get_forex_data(self, 
                      ticker: str = 'C:EURUSD',
                      timespan: str = 'minute',
                      from_date: Optional[str] = None,
                      to_date: Optional[str] = None,
                      limit: int = 50000) -> pd.DataFrame:
        """Specialized method for forex data"""
        return self.get_data(ticker, timespan, from_date, to_date, limit)