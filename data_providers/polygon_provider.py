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

    def test_connection(self) -> tuple[bool, str]:
        """
        Test the API connection with a simple request
        Returns: (success: bool, message: str)
        """
        try:
            # Use a simple request to get recent data for a common ticker
            test_ticker = "AAPL"
            today = datetime.now().strftime('%Y-%m-%d')
            yesterday = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            url = (f"{self.base_url}/{test_ticker}/range/1/day/{yesterday}/{today}"
                   f"?adjusted=true&sort=asc&limit=5&apiKey={self.api_key}")

            response = requests.get(url, timeout=10)

            # Check for authentication/authorization errors
            if response.status_code == 401:
                return False, "Invalid API key - Authentication failed"
            elif response.status_code == 403:
                return False, "Access forbidden - Check API key permissions"
            elif response.status_code == 429:
                return False, "Rate limit exceeded - API key may be invalid or overused"
            elif response.status_code != 200:
                return False, f"API request failed with status code {response.status_code}"

            # Parse response
            data = response.json()

            # Use the same validation logic as get_data() method
            if self.validate_response(data):
                # Valid response with results
                return True, "API key validated successfully"
            elif data.get('status') == 'ERROR':
                # API returned an error
                error_msg = data.get('error', data.get('message', 'Unknown error'))
                return False, f"API error: {error_msg}"
            elif data.get('status') == 'OK' and data.get('resultsCount', 0) == 0:
                # Valid API key but no data for this period (e.g., weekend)
                # This still means the key is valid
                return True, "API key validated successfully"
            else:
                # Unexpected response format
                return False, f"Unexpected API response format. Status: {data.get('status', 'none')}"

        except requests.exceptions.Timeout:
            return False, "Connection timeout - Check your internet connection"
        except requests.exceptions.ConnectionError:
            return False, "Connection error - Check your internet connection"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"