import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional
from .base_provider import BaseDataProvider

# Forex works exactly like crypto using the same REST API - no special client needed


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

        # Forex pairs work exactly like crypto - use the same REST API approach

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
        """Specialized method for forex data - works exactly like crypto"""
        return self.get_data(ticker, timespan, from_date, to_date, limit)


    def get_available_forex_pairs(self) -> list:
        """Get list of available forex pairs"""
        common_pairs = [
            'C:EURUSD',  # Euro / US Dollar
            'C:GBPUSD',  # British Pound / US Dollar
            'C:USDJPY',  # US Dollar / Japanese Yen
            'C:USDCHF',  # US Dollar / Swiss Franc
            'C:AUDUSD',  # Australian Dollar / US Dollar
            'C:USDCAD',  # US Dollar / Canadian Dollar
            'C:NZDUSD',  # New Zealand Dollar / US Dollar
            'C:EURGBP',  # Euro / British Pound
            'C:EURJPY',  # Euro / Japanese Yen
            'C:GBPJPY',  # British Pound / Japanese Yen
            'C:CHFJPY',  # Swiss Franc / Japanese Yen
            'C:EURCHF',  # Euro / Swiss Franc
            'C:AUDJPY',  # Australian Dollar / Japanese Yen
            'C:CADJPY',  # Canadian Dollar / Japanese Yen
            'C:NZDJPY',  # New Zealand Dollar / Japanese Yen
        ]
        return common_pairs

    def is_forex_pair(self, ticker: str) -> bool:
        """Check if ticker is a forex pair"""
        return ticker.startswith('C:') and len(ticker) == 8  # Format: C:EURUSD