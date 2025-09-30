import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
from typing import Optional
from datetime import datetime
from data_providers.base_provider import BaseDataProvider
import time


class OandaProvider(BaseDataProvider):
    """OANDA data provider for forex data"""

    def __init__(self, access_token: str, account_id: str, environment: str = "practice"):
        """
        Initialize OANDA provider

        Args:
            access_token: OANDA API access token
            account_id: OANDA account ID
            environment: 'practice' or 'live'
        """
        super().__init__(access_token)
        self.account_id = account_id
        self.environment = environment
        self.api = oandapyV20.API(access_token=access_token, environment=environment)

    def get_data(self,
                 ticker: str,
                 timespan: str = 'M1',
                 from_date: str = None,
                 to_date: str = None,
                 limit: int = 100) -> pd.DataFrame:
        """
        Get historical candle data from OANDA

        Args:
            ticker: Currency pair (e.g., 'EUR_USD')
            timespan: Granularity (M1, M5, M15, H1, D, etc.)
            from_date: Start date (ISO format)
            to_date: End date (ISO format)
            limit: Number of candles to retrieve (max 5000)

        Returns:
            DataFrame with OHLCV data and timestamp
        """
        # Convert ticker format (EURUSD -> EUR_USD)
        if '_' not in ticker:
            ticker = f"{ticker[:3]}_{ticker[3:]}"

        params = {
            "count": min(limit, 5000),
            "granularity": timespan
        }

        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        r = instruments.InstrumentsCandles(instrument=ticker, params=params)

        try:
            response = self.api.request(r)
            candles = response['candles']

            if not candles:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for candle in candles:
                data.append({
                    'timestamp': pd.to_datetime(candle['time']),
                    'Open': float(candle['mid']['o']),
                    'High': float(candle['mid']['h']),
                    'Low': float(candle['mid']['l']),
                    'Close': float(candle['mid']['c']),
                    'Volume': int(candle['volume']),
                    'complete': candle['complete']
                })

            df = pd.DataFrame(data)
            return df

        except oandapyV20.exceptions.V20Error as e:
            print(f"OANDA API Error: {e}")
            return pd.DataFrame()

    def get_live_data(self, ticker: str, lookback: int = 100) -> pd.DataFrame:
        """
        Get live data with recent candles for strategy

        Args:
            ticker: Currency pair (e.g., 'EUR_USD' or 'EURUSD')
            lookback: Number of recent candles to retrieve

        Returns:
            DataFrame with current OHLCV data
        """
        return self.get_data(ticker, timespan='M1', limit=lookback)

    def get_latest_candle(self, ticker: str) -> dict:
        """
        Get just the latest candle

        Args:
            ticker: Currency pair

        Returns:
            Dictionary with latest candle data
        """
        # Convert ticker format (EURUSD -> EUR_USD)
        if '_' not in ticker:
            ticker = f"{ticker[:3]}_{ticker[3:]}"

        params = {
            "count": 1,
            "granularity": "M1"
        }

        r = instruments.InstrumentsCandles(instrument=ticker, params=params)

        try:
            response = self.api.request(r)
            latest_candle = response['candles'][0]

            return {
                'time': latest_candle['time'],
                'open': float(latest_candle['mid']['o']),
                'high': float(latest_candle['mid']['h']),
                'low': float(latest_candle['mid']['l']),
                'close': float(latest_candle['mid']['c']),
                'volume': int(latest_candle['volume']),
                'complete': latest_candle['complete']
            }

        except oandapyV20.exceptions.V20Error as e:
            print(f"OANDA API Error: {e}")
            return {}

    def stream_prices(self, ticker: str, callback, duration: int = None):
        """
        Stream live prices (for future implementation)

        Args:
            ticker: Currency pair
            callback: Function to call with each price update
            duration: How long to stream (None for infinite)
        """
        # This would use OANDA's streaming API
        # For now, we'll use polling in the live trading engine
        pass