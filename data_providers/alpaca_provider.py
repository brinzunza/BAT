import requests
import json
import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from typing import Optional
from .base_provider import BaseDataProvider


class AlpacaDataProvider(BaseDataProvider):
    """Alpaca data provider for crypto and stock data"""

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

    def _is_crypto(self, ticker: str) -> bool:
        """Determine if ticker is cryptocurrency"""
        return '/' in ticker or ticker.upper().endswith('USD')

    def _get_data_endpoint(self, ticker: str) -> str:
        """Get appropriate data endpoint based on asset type"""
        if self._is_crypto(ticker):
            return f"{self.base_url}/v1beta3/crypto/us/bars"
        else:
            return f"{self.base_url}/v2/stocks/bars"

    def get_data(self,
                 ticker: str,
                 timespan: str = '1Min',
                 from_date: str = None,
                 to_date: str = None,
                 limit: int = 1000) -> pd.DataFrame:
        """Get historical data from Alpaca (crypto or stocks)"""

        is_crypto = self._is_crypto(ticker)

        # Convert ticker format
        if is_crypto and '/' in ticker:
            symbol = ticker.replace('/', '')
        else:
            symbol = ticker

        # Set default dates if not provided
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        # Get appropriate endpoint
        url = self._get_data_endpoint(ticker)

        # Set parameters based on asset type
        if is_crypto:
            params = {
                'symbols': symbol,
                'timeframe': timespan,
                'start': from_date,
                'end': to_date,
                'limit': limit,
                'sort': 'asc'
            }
        else:
            # Stock parameters - use the working format from your example
            params = {
                'symbols': symbol,
                'timeframe': timespan,
                'start': from_date,
                'end': to_date,
                'limit': limit,
                'adjustment': 'raw',
                'feed': 'sip',
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
            print(f"Error fetching historical data for {ticker}: {e}")
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

        is_crypto = self._is_crypto(ticker)

        if is_crypto:
            # Convert ticker format for the URL
            if '/' in ticker:
                symbol = ticker.replace('/', '%2F')  # URL encode the slash
            else:
                symbol = f"{ticker}%2FUSD"  # Assume USD pair if no slash

            # Use the public crypto endpoint (no authentication required)
            url = f"https://data.alpaca.markets/v1beta3/crypto/us/latest/bars?symbols={symbol}"
            symbol_key = ticker if '/' in ticker else f"{ticker}/USD"
        else:
            # For stocks, use the stock endpoint with query parameters
            symbol = ticker
            url = f"https://data.alpaca.markets/v2/stocks/bars/latest?symbols={symbol}"
            symbol_key = ticker

        try:
            if is_crypto:
                response = requests.get(url, headers={"accept": "application/json"})
            else:
                response = requests.get(url, headers=self.headers)

            response.raise_for_status()
            data = response.json()

            if is_crypto:
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
            else:
                # Stock data format - similar to crypto, should have bars[symbol]
                if 'bars' in data and symbol_key in data['bars']:
                    bar_data = data['bars'][symbol_key]
                    import pandas as pd
                    return {
                        'timestamp': pd.to_datetime(datetime.utcnow()),
                        'Open': float(bar_data['o']),
                        'High': float(bar_data['h']),
                        'Low': float(bar_data['l']),
                        'Close': float(bar_data['c']),
                        'Volume': float(bar_data['v'])
                    }
                else:
                    print(f"No bar data found for {symbol} in response")
                    print(f"Response keys: {list(data.keys())}")
                    if 'bars' in data:
                        print(f"Available symbols: {list(data['bars'].keys())}")
                    return {}

        except Exception as e:
            print(f"Error fetching latest bar for {ticker}: {e}")
            return {}

    def get_recent_bars_public(self, ticker: str, limit: int = 50) -> pd.DataFrame:
        """Get the last X bars using public endpoint for fast initialization"""
        from datetime import datetime, timedelta
        import pandas as pd

        is_crypto = self._is_crypto(ticker)

        # Calculate time range - get last 3 hours to ensure we have enough recent data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=3)

        # Format times for API (ISO format with Z)
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        if is_crypto:
            # Convert ticker format for the URL
            if '/' in ticker:
                symbol = ticker.replace('/', '%2F')  # URL encode the slash
            else:
                symbol = f"{ticker}%2FUSD"  # Assume USD pair if no slash

            # Use the public crypto bars endpoint (no authentication required)
            url = f"https://data.alpaca.markets/v1beta3/crypto/us/bars?symbols={symbol}&timeframe=1Min&start={start_str}&end={end_str}&limit={limit}&sort=desc"
            symbol_key = ticker if '/' in ticker else f"{ticker}/USD"
        else:
            # For stocks, use authenticated endpoint with proper parameters
            symbol = ticker
            url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={symbol}&timeframe=1Min&start={start_str}&end={end_str}&limit={limit}&adjustment=raw&feed=sip&sort=desc"
            symbol_key = ticker

        print(f"🔍 Fetching {ticker} data from: {start_str} to {end_str}")

        try:
            if is_crypto:
                response = requests.get(url, headers={"accept": "application/json"})
            else:
                response = requests.get(url, headers=self.headers)

            response.raise_for_status()
            data = response.json()

            print(f"📡 API Response keys: {list(data.keys())}")

            if is_crypto:
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
                else:
                    print(f"❌ No bars data found for {symbol_key} in response")
                    if 'bars' in data:
                        print(f"Available symbols in response: {list(data['bars'].keys())}")
                    else:
                        print(f"No 'bars' key in response. Keys: {list(data.keys())}")
                    return pd.DataFrame()
            else:
                # Stock data format - similar to crypto, symbol is in bars dict
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
                else:
                    print(f"❌ No bars data found for {symbol_key} in response")
                    if 'bars' in data:
                        print(f"Available symbols in response: {list(data['bars'].keys())}")
                    else:
                        print(f"No 'bars' key in response. Keys: {list(data.keys())}")
                    return pd.DataFrame()

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

        except Exception as e:
            print(f"❌ Error fetching recent bars for {ticker}: {e}")
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

    def buy(self, symbol: str, quantity: float, order_type: str = "market", limit_price: float = None) -> dict:
        """Place buy order using alpaca_trade_api library"""

        original_symbol = symbol
        # Convert symbol format for crypto (BTC/USD -> BTCUSD), keep stocks as-is
        if '/' in symbol:
            symbol = symbol.replace('/', '')

        print(f"🔄 Placing buy order: {quantity} {original_symbol} using alpaca_trade_api")

        try:
            # Use the official alpaca_trade_api library
            order_params = {
                'symbol': symbol,
                'qty': quantity,
                'side': 'buy',
                'type': order_type,
                'time_in_force': 'gtc'
            }

            # Add limit price for limit orders
            if order_type == "limit" and limit_price is not None:
                order_params['limit_price'] = limit_price
                print(f"💰 Limit price set to: ${limit_price:.2f}")

            order = self.trade_api.submit_order(**order_params)

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

    def sell(self, symbol: str, quantity: float, order_type: str = "market", limit_price: float = None) -> dict:
        """Place sell order using alpaca_trade_api library"""

        original_symbol = symbol
        # Convert symbol format for crypto (BTC/USD -> BTCUSD), keep stocks as-is
        if '/' in symbol:
            symbol = symbol.replace('/', '')

        print(f"🔄 Placing sell order: {quantity} {original_symbol} using alpaca_trade_api")

        try:
            # Use the official alpaca_trade_api library
            order_params = {
                'symbol': symbol,
                'qty': quantity,
                'side': 'sell',
                'type': order_type,
                'time_in_force': 'gtc'
            }

            # Add limit price for limit orders
            if order_type == "limit" and limit_price is not None:
                order_params['limit_price'] = limit_price
                print(f"💰 Limit price set to: ${limit_price:.2f}")

            order = self.trade_api.submit_order(**order_params)

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

    def get_positions_api(self):
        """Get all positions using direct API call"""
        try:
            url = f"{self.base_url}/v2/positions"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting positions via API: {e}")
            return []

    def get_position_for_symbol(self, symbol: str):
        """Get position for specific symbol using direct API call"""
        try:
            # Convert symbol format (BTC/USD -> BTCUSD)
            if '/' in symbol:
                symbol = symbol.replace('/', '')

            url = f"{self.base_url}/v2/positions/{symbol}"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 404:
                # No position exists
                return {
                    'symbol': symbol,
                    'qty': '0',
                    'side': 'long',
                    'avg_entry_price': '0',
                    'market_value': '0',
                    'unrealized_pl': '0',
                    'unrealized_plpc': '0'
                }

            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting position for {symbol}: {e}")
            return {
                'symbol': symbol,
                'qty': '0',
                'side': 'long',
                'avg_entry_price': '0',
                'market_value': '0',
                'unrealized_pl': '0',
                'unrealized_plpc': '0'
            }

    def get_account_api(self):
        """Get account information using direct API call"""
        try:
            url = f"{self.base_url}/v2/account"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting account via API: {e}")
            return {}

    def get_portfolio_history(self, period: str = "1D"):
        """Get portfolio history using direct API call"""
        try:
            url = f"{self.base_url}/v2/account/portfolio/history"
            params = {
                'period': period,
                'timeframe': '1Min'
            }
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting portfolio history: {e}")
            return {}

    def get_orders_api(self, status: str = "all", limit: int = 50):
        """Get orders using direct API call"""
        try:
            url = f"{self.base_url}/v2/orders"
            params = {
                'status': status,
                'limit': limit,
                'direction': 'desc'
            }
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting orders via API: {e}")
            return []

    def close_position(self, symbol: str, cancel_orders: bool = True):
        """Close position for specific symbol using direct API call"""
        try:
            # Convert symbol format (BTC/USD -> BTCUSD)
            if '/' in symbol:
                symbol = symbol.replace('/', '')

            url = f"{self.base_url}/v2/positions/{symbol}"
            params = {
                'cancel_orders': str(cancel_orders).lower()
            }
            response = requests.delete(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error closing position for {symbol}: {e}")
            return {'status': 'failed', 'error': str(e)}

    def close_all_positions(self, cancel_orders: bool = True):
        """Close all positions using direct API call"""
        try:
            url = f"{self.base_url}/v2/positions"
            params = {
                'cancel_orders': str(cancel_orders).lower()
            }
            response = requests.delete(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error closing all positions: {e}")
            return {'status': 'failed', 'error': str(e)}

    def cancel_order(self, order_id: str):
        """Cancel a specific order using direct API call"""
        try:
            url = f"{self.base_url}/v2/orders/{order_id}"
            response = requests.delete(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error canceling order {order_id}: {e}")
            return {'status': 'failed', 'error': str(e)}


class SimulatedBroker:
    """Simulated broker that uses live Alpaca data but manages local account"""

    def __init__(self, api_key: str, secret_key: str, initial_balance: float = 10000):
        self.api_key = api_key
        self.secret_key = secret_key
        self.initial_balance = initial_balance

        # Initialize data provider for live prices
        self.data_provider = AlpacaDataProvider(api_key, secret_key)

        # Local account state
        self.reset_account()

        # Order management
        self.order_counter = 1

    def reset_account(self):
        """Reset the simulated account to initial state"""
        self.cash_balance = self.initial_balance
        self.positions = {}  # symbol -> {'qty': float, 'avg_entry_price': float, 'side': str}
        self.orders = {}  # order_id -> order details
        self.trade_history = []

    def get_current_price(self, symbol: str) -> float:
        """Get current market price from live Alpaca data"""
        try:
            original_symbol = symbol

            # Get data using original symbol format (the data provider handles conversion)
            latest_bar = self.data_provider.get_latest_bar(original_symbol)
            if latest_bar:
                return latest_bar['Close']
            else:
                # Fallback to quote data
                quote_data = self.data_provider.get_latest_quote(original_symbol)
                return quote_data.get('close', 0)
        except Exception as e:
            print(f"⚠️ Error getting current price for {symbol}: {e}")
            return 0

    def _generate_order_id(self) -> str:
        """Generate a unique order ID"""
        order_id = f"SIM_{self.order_counter:06d}"
        self.order_counter += 1
        return order_id

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions)"""
        total_value = self.cash_balance

        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            position_value = float(position['qty']) * current_price
            total_value += position_value

        return total_value

    def _calculate_unrealized_pnl(self, symbol: str) -> float:
        """Calculate unrealized P&L for a position"""
        if symbol not in self.positions:
            return 0

        position = self.positions[symbol]
        qty = float(position['qty'])
        avg_entry = float(position['avg_entry_price'])
        current_price = self.get_current_price(symbol)

        if qty > 0:  # Long position
            return (current_price - avg_entry) * qty
        elif qty < 0:  # Short position
            return (avg_entry - current_price) * abs(qty)
        else:
            return 0

    def buy(self, symbol: str, quantity: float, order_type: str = "market", limit_price: float = None) -> dict:
        """Simulate buy order execution"""
        original_symbol = symbol

        # Use original symbol for price lookups, but normalize for internal storage
        storage_symbol = symbol.replace('/', '') if '/' in symbol else symbol

        order_id = self._generate_order_id()
        current_price = self.get_current_price(original_symbol)

        if current_price <= 0:
            return {'status': 'failed', 'error': 'Unable to get current price'}

        # For limit orders, check if price is acceptable
        if order_type == "limit" and limit_price is not None:
            execution_price = limit_price
            if current_price > limit_price:  # Current price too high for buy limit
                # Store as pending order
                self.orders[order_id] = {
                    'id': order_id,
                    'symbol': symbol,
                    'qty': quantity,
                    'side': 'buy',
                    'type': order_type,
                    'limit_price': limit_price,
                    'status': 'pending',
                    'created_at': datetime.now()
                }
                print(f"📋 BUY LIMIT ORDER PENDING - {quantity} {symbol} @ ${limit_price:.2f} (current: ${current_price:.2f})")
                return {'id': order_id, 'status': 'pending', 'symbol': symbol, 'qty': quantity}
        else:
            execution_price = current_price

        # Calculate trade value
        trade_value = quantity * execution_price

        # Check buying power
        if trade_value > self.cash_balance:
            return {'status': 'failed', 'error': 'Insufficient buying power'}

        # Execute the trade
        self._execute_buy(storage_symbol, quantity, execution_price, order_id)

        print(f"✅ SIMULATED BUY EXECUTED - {quantity} {original_symbol} @ ${execution_price:.2f}")
        print(f"💰 Cash Balance: ${self.cash_balance:.2f}")

        return {
            'id': order_id,
            'status': 'filled',
            'symbol': original_symbol,
            'qty': quantity,
            'filled_price': execution_price
        }

    def _execute_buy(self, symbol: str, quantity: float, price: float, order_id: str):
        """Execute a buy order"""
        trade_value = quantity * price

        # Update cash balance
        self.cash_balance -= trade_value

        # Update position
        if symbol in self.positions:
            existing_qty = float(self.positions[symbol]['qty'])
            existing_avg = float(self.positions[symbol]['avg_entry_price'])

            if existing_qty < 0:  # Closing short position
                if quantity >= abs(existing_qty):
                    # Close short and potentially open long
                    remaining_qty = quantity - abs(existing_qty)
                    self.positions[symbol] = {
                        'qty': remaining_qty,
                        'avg_entry_price': price,
                        'side': 'long' if remaining_qty > 0 else 'flat'
                    }
                else:
                    # Partially close short
                    self.positions[symbol]['qty'] = existing_qty + quantity
            else:  # Adding to long position
                total_qty = existing_qty + quantity
                new_avg = ((existing_qty * existing_avg) + (quantity * price)) / total_qty
                self.positions[symbol] = {
                    'qty': total_qty,
                    'avg_entry_price': new_avg,
                    'side': 'long'
                }
        else:
            # New position
            self.positions[symbol] = {
                'qty': quantity,
                'avg_entry_price': price,
                'side': 'long'
            }

        # Record trade
        self.trade_history.append({
            'order_id': order_id,
            'symbol': symbol,
            'side': 'buy',
            'qty': quantity,
            'price': price,
            'timestamp': datetime.now()
        })

    def sell(self, symbol: str, quantity: float, order_type: str = "market", limit_price: float = None) -> dict:
        """Simulate sell order execution"""
        original_symbol = symbol

        # Use original symbol for price lookups, but normalize for internal storage
        storage_symbol = symbol.replace('/', '') if '/' in symbol else symbol

        order_id = self._generate_order_id()
        current_price = self.get_current_price(original_symbol)

        if current_price <= 0:
            return {'status': 'failed', 'error': 'Unable to get current price'}

        # For limit orders, check if price is acceptable
        if order_type == "limit" and limit_price is not None:
            execution_price = limit_price
            if current_price < limit_price:  # Current price too low for sell limit
                # Store as pending order
                self.orders[order_id] = {
                    'id': order_id,
                    'symbol': symbol,
                    'qty': quantity,
                    'side': 'sell',
                    'type': order_type,
                    'limit_price': limit_price,
                    'status': 'pending',
                    'created_at': datetime.now()
                }
                print(f"📋 SELL LIMIT ORDER PENDING - {quantity} {symbol} @ ${limit_price:.2f} (current: ${current_price:.2f})")
                return {'id': order_id, 'status': 'pending', 'symbol': symbol, 'qty': quantity}
        else:
            execution_price = current_price

        # Execute the trade
        self._execute_sell(storage_symbol, quantity, execution_price, order_id)

        print(f"✅ SIMULATED SELL EXECUTED - {quantity} {original_symbol} @ ${execution_price:.2f}")
        print(f"💰 Cash Balance: ${self.cash_balance:.2f}")

        return {
            'id': order_id,
            'status': 'filled',
            'symbol': original_symbol,
            'qty': quantity,
            'filled_price': execution_price
        }

    def _execute_sell(self, symbol: str, quantity: float, price: float, order_id: str):
        """Execute a sell order"""
        trade_value = quantity * price

        # Update cash balance
        self.cash_balance += trade_value

        # Update position
        if symbol in self.positions:
            existing_qty = float(self.positions[symbol]['qty'])
            existing_avg = float(self.positions[symbol]['avg_entry_price'])

            if existing_qty > 0:  # Closing long position
                if quantity >= existing_qty:
                    # Close long and potentially open short
                    remaining_qty = quantity - existing_qty
                    self.positions[symbol] = {
                        'qty': -remaining_qty,
                        'avg_entry_price': price,
                        'side': 'short' if remaining_qty > 0 else 'flat'
                    }
                else:
                    # Partially close long
                    self.positions[symbol]['qty'] = existing_qty - quantity
            else:  # Adding to short position
                total_qty = abs(existing_qty) + quantity
                new_avg = ((abs(existing_qty) * existing_avg) + (quantity * price)) / total_qty
                self.positions[symbol] = {
                    'qty': -total_qty,
                    'avg_entry_price': new_avg,
                    'side': 'short'
                }
        else:
            # New short position
            self.positions[symbol] = {
                'qty': -quantity,
                'avg_entry_price': price,
                'side': 'short'
            }

        # Record trade
        self.trade_history.append({
            'order_id': order_id,
            'symbol': symbol,
            'side': 'sell',
            'qty': quantity,
            'price': price,
            'timestamp': datetime.now()
        })

    def close_position(self, symbol: str) -> dict:
        """Close all positions for a symbol using market orders"""
        # Convert symbol format
        if '/' in symbol:
            symbol = symbol.replace('/', '')

        if symbol not in self.positions:
            return {'status': 'failed', 'error': 'No position to close'}

        position = self.positions[symbol]
        qty = float(position['qty'])

        if qty == 0:
            return {'status': 'failed', 'error': 'No position to close'}

        # Close position with market order
        if qty > 0:  # Close long position
            return self.sell(symbol, qty, order_type="market")
        else:  # Close short position
            return self.buy(symbol, abs(qty), order_type="market")

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending order"""
        if order_id not in self.orders:
            return {'status': 'failed', 'error': 'Order not found'}

        order = self.orders[order_id]
        if order['status'] != 'pending':
            return {'status': 'failed', 'error': 'Order is not pending'}

        # Remove from pending orders
        del self.orders[order_id]
        print(f"✅ SIMULATED ORDER CANCELED - {order['side'].upper()} {order['qty']} {order['symbol']}")

        return {'status': 'canceled', 'order_id': order_id}

    def check_pending_orders(self):
        """Check if any pending limit orders can be filled based on current prices"""
        filled_orders = []

        for order_id, order in list(self.orders.items()):
            if order['status'] == 'pending':
                symbol = order['symbol']
                current_price = self.get_current_price(symbol)

                should_fill = False
                if order['side'] == 'buy' and current_price <= order['limit_price']:
                    should_fill = True
                elif order['side'] == 'sell' and current_price >= order['limit_price']:
                    should_fill = True

                if should_fill:
                    print(f"🎯 LIMIT ORDER TRIGGERED - {order['side'].upper()} {order['qty']} {symbol} @ ${order['limit_price']:.2f}")

                    # Execute the order
                    if order['side'] == 'buy':
                        self._execute_buy(symbol, order['qty'], order['limit_price'], order_id)
                    else:
                        self._execute_sell(symbol, order['qty'], order['limit_price'], order_id)

                    # Update order status
                    order['status'] = 'filled'
                    filled_orders.append(order_id)

        # Remove filled orders from pending
        for order_id in filled_orders:
            if order_id in self.orders:
                del self.orders[order_id]

        return len(filled_orders)

    def get_position_for_symbol(self, symbol: str) -> dict:
        """Get position information for a specific symbol"""
        original_symbol = symbol
        # Normalize symbol for internal storage
        storage_symbol = symbol.replace('/', '') if '/' in symbol else symbol

        if storage_symbol not in self.positions:
            return {
                'symbol': symbol,
                'qty': '0',
                'side': 'long',
                'avg_entry_price': '0',
                'market_value': '0',
                'unrealized_pl': '0',
                'unrealized_plpc': '0'
            }

        position = self.positions[storage_symbol]
        qty = float(position['qty'])
        avg_entry = float(position['avg_entry_price'])
        current_price = self.get_current_price(original_symbol)

        market_value = qty * current_price
        unrealized_pnl = self._calculate_unrealized_pnl(storage_symbol)
        unrealized_plpc = (unrealized_pnl / (abs(qty) * avg_entry)) * 100 if qty != 0 and avg_entry != 0 else 0

        return {
            'symbol': original_symbol,
            'qty': str(qty),
            'side': position['side'],
            'avg_entry_price': str(avg_entry),
            'market_value': str(market_value),
            'unrealized_pl': str(unrealized_pnl),
            'unrealized_plpc': str(unrealized_plpc)
        }

    def get_account(self) -> dict:
        """Get simulated account information"""
        portfolio_value = self._calculate_portfolio_value()
        equity = portfolio_value

        # Calculate total unrealized P&L
        total_unrealized_pnl = sum(self._calculate_unrealized_pnl(symbol) for symbol in self.positions.keys())

        return {
            'id': 'simulated_account',
            'status': 'ACTIVE',
            'buying_power': str(self.cash_balance),
            'equity': str(equity),
            'portfolio_value': str(portfolio_value),
            'cash': str(self.cash_balance),
            'initial_balance': str(self.initial_balance),
            'total_unrealized_pnl': str(total_unrealized_pnl),
            'trading_blocked': False,
            'account_blocked': False,
            'pattern_day_trader': False
        }

    def get_account_api(self) -> dict:
        """Alias for get_account() to match AlpacaBroker interface"""
        return self.get_account()

    def print_account_summary(self):
        """Print a summary of the simulated account"""
        account = self.get_account()
        print(f"\n{'='*50}")
        print(f"📊 SIMULATED ACCOUNT SUMMARY")
        print(f"{'='*50}")
        print(f"💰 Cash Balance: ${float(account['cash']):,.2f}")
        print(f"📈 Portfolio Value: ${float(account['portfolio_value']):,.2f}")
        print(f"💵 Equity: ${float(account['equity']):,.2f}")
        print(f"💲 Total P&L: ${float(account['equity']) - self.initial_balance:,.2f}")
        print(f"📊 Return: {((float(account['equity']) / self.initial_balance) - 1) * 100:.2f}%")

        if self.positions:
            print(f"\n📋 POSITIONS:")
            for symbol, position in self.positions.items():
                if float(position['qty']) != 0:
                    unrealized_pnl = self._calculate_unrealized_pnl(symbol)
                    current_price = self.get_current_price(symbol)
                    print(f"  {symbol}: {position['qty']} @ ${position['avg_entry_price']:.2f} (current: ${current_price:.2f}) | P&L: ${unrealized_pnl:.2f}")

        if self.orders:
            print(f"\n📝 PENDING ORDERS:")
            for order_id, order in self.orders.items():
                print(f"  {order_id}: {order['side'].upper()} {order['qty']} {order['symbol']} @ ${order['limit_price']:.2f}")

        print(f"{'='*50}")