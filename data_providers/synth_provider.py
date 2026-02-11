import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional
from .base_provider import BaseDataProvider


class SynthDataProvider(BaseDataProvider):
    """Synth data provider for real-time synthetic market data"""

    def __init__(self, base_url: str = "http://35.209.219.174:8000", api_key: str = ""):
        # Synth API requires authentication via query parameter
        if not api_key:
            raise ValueError("API key is required for Synth provider")
        super().__init__(api_key=api_key)
        self.base_url = base_url.rstrip('/')

    def get_live_data(self, ticker: str = 'SYNTH') -> pd.DataFrame:
        """
        Get live/current data for a ticker from the Synth API

        Args:
            ticker: The ticker symbol (default: 'SYNTH')

        Returns:
            DataFrame with current OHLCV data and timestamp
        """
        # Use lowercase ticker for API endpoint
        ticker_lower = ticker.lower()

        # Build URL for candles endpoint (1-minute interval)
        url = f"{self.base_url}/candles/{ticker_lower}/1m?api_key={self.api_key}"

        try:
            response = requests.get(url, timeout=5)

            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

            data = response.json()

            # Parse the response format:
            # {"symbol":"SYNTH","interval":"1m","candle":{"timestamp":1770834300,"open":243.02,
            #  "high":248.44,"low":241.26,"close":248.44,"volume":317241}}

            candle = data['candle']

            # Convert to standardized OHLCV format
            df = pd.DataFrame([{
                'timestamp': pd.to_datetime(candle['timestamp'], unit='s'),
                'Open': candle['open'],
                'High': candle['high'],
                'Low': candle['low'],
                'Close': candle['close'],
                'Volume': candle['volume']
            }])

            return df

        except requests.exceptions.Timeout:
            raise Exception("Connection timeout - Check if Synth API is reachable")
        except requests.exceptions.ConnectionError:
            raise Exception("Connection error - Check if Synth API is running")
        except KeyError as e:
            raise Exception(f"Invalid API response format - missing field: {e}")
        except Exception as e:
            raise Exception(f"Failed to fetch live data: {str(e)}")

    def get_data(self,
                 ticker: str = 'SYNTH',
                 timespan: str = 'minute',
                 from_date: Optional[str] = None,
                 to_date: Optional[str] = None,
                 limit: int = 50000) -> pd.DataFrame:
        """
        Get historical data for a ticker

        Note: The Synth API currently only provides real-time data.
        This method simulates historical data by calling get_live_data()
        repeatedly with a small delay to build a time series.

        For true historical data support, the Synth API would need to provide
        a historical endpoint.

        Args:
            ticker: The ticker symbol
            timespan: Time interval (not used in current implementation)
            from_date: Start date (not used in current implementation)
            to_date: End date (not used in current implementation)
            limit: Maximum number of records (not used in current implementation)

        Returns:
            DataFrame with OHLCV data and timestamp
        """
        # For now, just return the latest data point
        # In a production environment, you'd want to either:
        # 1. Call a historical endpoint if available
        # 2. Store data locally and build history over time
        # 3. Use a time-series database to accumulate data

        return self.get_live_data(ticker)

    def get_latest_tick(self, ticker: str = 'SYNTH') -> dict:
        """
        Get the latest tick data as a dictionary

        Args:
            ticker: The ticker symbol

        Returns:
            Dictionary with all fields from the API response
        """
        # Use lowercase ticker for API endpoint
        ticker_lower = ticker.lower()

        # Build URL for candles endpoint
        url = f"{self.base_url}/candles/{ticker_lower}/1m?api_key={self.api_key}"

        try:
            response = requests.get(url, timeout=5)

            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

            return response.json()

        except Exception as e:
            raise Exception(f"Failed to fetch latest tick: {str(e)}")

    def test_connection(self) -> tuple[bool, str]:
        """
        Test the API connection
        Returns: (success: bool, message: str)
        """
        try:
            # Try to fetch data for the default SYNTH ticker
            df = self.get_live_data('SYNTH')

            if df is not None and not df.empty:
                return True, "Synth API connection validated successfully"
            else:
                return False, "Synth API returned empty data"

        except Exception as e:
            return False, f"Connection test failed: {str(e)}"

    def validate_response(self, data: dict) -> bool:
        """
        Validate API response has required fields

        Args:
            data: API response dictionary

        Returns:
            True if response has all required fields
        """
        # New format validation
        if 'candle' in data:
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            return all(field in data['candle'] for field in required_fields)

        # Old format validation (legacy support)
        required_fields = ['symbol', 'price', 'open', 'high', 'low', 'volume', 'updated_at']
        return all(field in data for field in required_fields)
