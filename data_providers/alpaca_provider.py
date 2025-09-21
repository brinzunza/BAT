import requests
import json
import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from typing import Optional
from .base_provider import BaseDataProvider


class AlpacaDataProvider(BaseDataProvider):
    """Alpaca data provider for crypto data"""

    def __init__(self, api_key: str = None, secret_key: str = None):
        super().__init__(api_key)
        self.secret_key = secret_key
        self.base_url = "https://data.alpaca.markets"
        self.paper_base_url = "https://paper-api.alpaca.markets"
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key
        }

    def get_data(self,
                 ticker: str,
                 timespan: str = '1Min',
                 from_date: str = None,
                 to_date: str = None,
                 limit: int = 1000) -> pd.DataFrame:
        """Get historical crypto data from Alpaca"""

        # Convert ticker format for crypto
        if '/' in ticker:
            symbol = ticker.replace('/', '')
        else:
            symbol = ticker

        # Set default dates if not provided
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        url = f"{self.base_url}/v1beta3/crypto/us/bars"
        params = {
            'symbols': symbol,
            'timeframe': timespan,
            'start': from_date,
            'end': to_date,
            'limit': limit,
            'sort': 'asc'
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            if 'bars' not in data or symbol not in data['bars']:
                return pd.DataFrame()

            # Convert to DataFrame
            bars = data['bars'][symbol]
            df = pd.DataFrame(bars)

            if df.empty:
                return df

            # Standardize column names
            df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                't': 'timestamp'
            }, inplace=True)

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Drop unnecessary columns
            df.drop(columns=['vw', 'n'], inplace=True, errors='ignore')

            return df

        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def get_live_data(self, ticker: str, lookback_minutes: int = 100) -> pd.DataFrame:
        """Get recent live data for crypto"""

        # Get recent historical data as "live" data
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=lookback_minutes)

        return self.get_data(
            ticker=ticker,
            timespan='1Min',
            from_date=start_time.strftime('%Y-%m-%d'),
            to_date=end_time.strftime('%Y-%m-%d'),
            limit=lookback_minutes
        )

    def get_latest_bar(self, ticker: str) -> dict:
        """Get only the latest bar for live trading using public endpoint"""

        # Convert ticker format for the URL
        if '/' in ticker:
            symbol = ticker.replace('/', '%2F')  # URL encode the slash
        else:
            symbol = f"{ticker}%2FUSD"  # Assume USD pair if no slash

        # Use the public endpoint (no authentication required)
        url = f"https://data.alpaca.markets/v1beta3/crypto/us/latest/bars?symbols={symbol}"

        try:
            response = requests.get(url, headers={"accept": "application/json"})
            response.raise_for_status()
            data = response.json()

            # Extract the symbol key (e.g., 'BTC/USD' or 'BTCUSD')
            symbol_key = ticker if '/' in ticker else f"{ticker}/USD"

            if 'bars' in data and symbol_key in data['bars']:
                bar_data = data['bars'][symbol_key]
                import pandas as pd
                return {
                    'timestamp': pd.to_datetime(datetime.utcnow()),  # Use timezone-aware UTC time
                    'Open': float(bar_data['o']),
                    'High': float(bar_data['h']),
                    'Low': float(bar_data['l']),
                    'Close': float(bar_data['c']),
                    'Volume': float(bar_data['v'])
                }
            else:
                print(f"No data found for {symbol_key} in response")
                return {}

        except Exception as e:
            print(f"Error fetching latest bar from public endpoint: {e}")
            return {}

    def get_recent_bars_public(self, ticker: str, limit: int = 50) -> pd.DataFrame:
        """Get the last X bars using public endpoint for fast initialization"""
        from datetime import datetime, timedelta
        import pandas as pd

        # Convert ticker format for the URL
        if '/' in ticker:
            symbol = ticker.replace('/', '%2F')  # URL encode the slash
        else:
            symbol = f"{ticker}%2FUSD"  # Assume USD pair if no slash

        # Calculate time range - get last 3 hours to ensure we have enough recent data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=3)

        # Format times for API (ISO format with Z)
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Use the public bars endpoint (no authentication required)
        url = f"https://data.alpaca.markets/v1beta3/crypto/us/bars?symbols={symbol}&timeframe=1Min&start={start_str}&end={end_str}&limit={limit}&sort=desc"

        print(f"🔍 Fetching data from: {start_str} to {end_str}")

        try:
            response = requests.get(url, headers={"accept": "application/json"})
            response.raise_for_status()
            data = response.json()

            # Extract the symbol key (e.g., 'BTC/USD')
            symbol_key = ticker if '/' in ticker else f"{ticker}/USD"

            print(f"📡 API Response keys: {list(data.keys())}")
            if 'bars' in data:
                print(f"📊 Available symbols: {list(data['bars'].keys())}")

            if 'bars' in data and symbol_key in data['bars']:
                bars_data = data['bars'][symbol_key]
                print(f"✅ Found {len(bars_data)} bars for {symbol_key}")

                if not bars_data:
                    print(f"⚠️ No bars data returned for {symbol_key}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df_data = []
                for bar in bars_data:
                    df_data.append({
                        'timestamp': pd.to_datetime(bar['t']),
                        'Open': float(bar['o']),
                        'High': float(bar['h']),
                        'Low': float(bar['l']),
                        'Close': float(bar['c']),
                        'Volume': float(bar['v'])
                    })

                df = pd.DataFrame(df_data)

                if df.empty:
                    print(f"⚠️ DataFrame is empty after conversion")
                    return df

                # Sort by timestamp (desc gives us newest first, so reverse to get chronological order)
                df = df.sort_values('timestamp').reset_index(drop=True)

                print(f"📊 Successfully fetched {len(df)} recent bars")
                print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"💰 Latest price: ${df['Close'].iloc[-1]:,.2f}")

                return df

            else:
                print(f"❌ No bars data found for {symbol_key} in response")
                if 'bars' in data:
                    print(f"Available symbols in response: {list(data['bars'].keys())}")
                else:
                    print(f"No 'bars' key in response. Keys: {list(data.keys())}")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Error fetching recent bars from public endpoint: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_latest_quote(self, ticker: str) -> dict:
        """Get latest quote data"""

        if '/' in ticker:
            symbol = ticker.replace('/', '')
        else:
            symbol = ticker

        url = f"{self.base_url}/v1beta3/crypto/us/latest/bars"
        params = {'symbols': symbol}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            if 'bars' in data and symbol in data['bars']:
                bar_data = data['bars'][symbol]
                return {
                    'timestamp': datetime.now(),
                    'open': float(bar_data['o']),
                    'high': float(bar_data['h']),
                    'low': float(bar_data['l']),
                    'close': float(bar_data['c']),
                    'volume': float(bar_data['v'])
                }
        except Exception as e:
            print(f"Error fetching latest quote: {e}")

        return {}


class AlpacaBroker:
    """Alpaca broker interface for placing trades"""

    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper_trading = paper_trading

        if paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"

        # Use the official alpaca_trade_api library
        self.trade_api = tradeapi.REST(api_key, secret_key, self.base_url, api_version='v2')

        # Keep headers for direct API calls when needed
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key
        }

    def get_account(self):
        """Get account information using alpaca_trade_api"""
        try:
            account = self.trade_api.get_account()
            # Convert to dict for consistency
            return {
                'id': account.id,
                'status': account.status,
                'buying_power': account.buying_power,
                'equity': account.equity,
                'portfolio_value': account.portfolio_value,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}

    def test_authentication(self):
        """Test if API credentials are working"""
        print(f"🔐 Testing Alpaca authentication...")
        print(f"📡 Base URL: {self.base_url}")
        print(f"🔑 API Key: {self.api_key[:8]}..." if self.api_key else "❌ No API Key")
        print(f"🔑 Secret Key: {'✅ Present' if self.secret_key else '❌ Missing'}")

        account = self.get_account()
        if account:
            print(f"✅ Authentication successful!")
            print(f"📊 Account ID: {account.get('id', 'Unknown')}")
            print(f"💰 Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
            print(f"📈 Account Status: {account.get('status', 'Unknown')}")

            # Check if crypto trading is enabled
            if 'crypto_status' in account:
                print(f"₿ Crypto Status: {account['crypto_status']}")
            else:
                print(f"₿ Crypto Status: Not found in account info")

            # Check trading permissions
            if 'trading_blocked' in account:
                print(f"🚫 Trading Blocked: {account['trading_blocked']}")

            if 'account_blocked' in account:
                print(f"🚫 Account Blocked: {account['account_blocked']}")

            # Check pattern day trader status
            if 'pattern_day_trader' in account:
                print(f"📊 PDT Status: {account['pattern_day_trader']}")

            return True
        else:
            print(f"❌ Authentication failed!")
            return False

    def check_crypto_permissions(self):
        """Check if crypto trading is enabled for this account"""
        print(f"🔍 Checking crypto trading permissions...")

        # Try to get crypto positions (this will fail if crypto is not enabled)
        try:
            url = f"{self.base_url}/v2/positions"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            positions = response.json()

            print(f"✅ Can access positions endpoint")

            # Check if any crypto positions exist
            crypto_positions = [p for p in positions if p.get('symbol', '').endswith('USD') and p.get('symbol', '') in ['BTCUSD', 'ETHUSD']]
            if crypto_positions:
                print(f"₿ Found {len(crypto_positions)} crypto positions")
            else:
                print(f"₿ No crypto positions found (this is normal)")

            return True

        except Exception as e:
            print(f"❌ Error checking crypto permissions: {e}")
            return False

    def buy(self, symbol: str, quantity: float, order_type: str = "market") -> dict:
        """Place buy order using alpaca_trade_api library"""

        # Convert symbol format for crypto (BTC/USD -> BTCUSD)
        if '/' in symbol:
            symbol = symbol.replace('/', '')

        print(f"🔄 Placing buy order: {quantity} {symbol} using alpaca_trade_api")

        try:
            # Use the official alpaca_trade_api library
            order = self.trade_api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type=order_type,
                time_in_force='gtc'
            )

            print(f"✅ Buy order placed successfully!")
            print(f"📋 Order ID: {order.id}")
            print(f"📊 Status: {order.status}")

            # Convert order object to dict for consistency
            result = {
                'id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'submitted_at': str(order.submitted_at)
            }

            return result

        except Exception as e:
            print(f"❌ Error placing buy order: {e}")
            print(f"📋 Error type: {type(e)}")

            # Try to get more detailed error information
            if hasattr(e, 'response'):
                print(f"📄 Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'Unknown'}")
                print(f"📄 Response text: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")

            return {}

    def sell(self, symbol: str, quantity: float, order_type: str = "market") -> dict:
        """Place sell order using alpaca_trade_api library"""

        # Convert symbol format for crypto (BTC/USD -> BTCUSD)
        if '/' in symbol:
            symbol = symbol.replace('/', '')

        print(f"🔄 Placing sell order: {quantity} {symbol} using alpaca_trade_api")

        try:
            # Use the official alpaca_trade_api library
            order = self.trade_api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type=order_type,
                time_in_force='gtc'
            )

            print(f"✅ Sell order placed successfully!")
            print(f"📋 Order ID: {order.id}")
            print(f"📊 Status: {order.status}")

            # Convert order object to dict for consistency
            result = {
                'id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'submitted_at': str(order.submitted_at)
            }

            return result

        except Exception as e:
            print(f"❌ Error placing sell order: {e}")
            print(f"📋 Error type: {type(e)}")

            # Try to get more detailed error information
            if hasattr(e, 'response'):
                print(f"📄 Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'Unknown'}")
                print(f"📄 Response text: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")

            return {}

    def get_positions(self):
        """Get current positions using alpaca_trade_api"""
        try:
            positions = self.trade_api.list_positions()
            # Convert to list of dicts for consistency
            return [
                {
                    'symbol': pos.symbol,
                    'qty': pos.qty,
                    'side': pos.side,
                    'avg_entry_price': pos.avg_entry_price,
                    'market_value': pos.market_value,
                    'unrealized_pl': pos.unrealized_pl
                }
                for pos in positions
            ]
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def get_orders(self, status: str = "open"):
        """Get orders using alpaca_trade_api"""
        try:
            orders = self.trade_api.list_orders(status=status)
            # Convert to list of dicts for consistency
            return [
                {
                    'id': order.id,
                    'symbol': order.symbol,
                    'qty': order.qty,
                    'side': order.side,
                    'type': order.type,
                    'status': order.status,
                    'submitted_at': str(order.submitted_at)
                }
                for order in orders
            ]
        except Exception as e:
            print(f"Error getting orders: {e}")
            return []

    def get_buying_power(self):
        """Get account buying power as initial balance"""
        account_info = self.get_account()
        if account_info and 'buying_power' in account_info:
            return float(account_info['buying_power'])
        return 0.0