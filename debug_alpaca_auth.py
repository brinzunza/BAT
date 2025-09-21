#!/usr/bin/env python3
"""
Non-interactive script to debug Alpaca API authentication and order placement issues
"""

import os
from data_providers.alpaca_provider import AlpacaBroker

def debug_alpaca_authentication():
    """Debug Alpaca authentication and order placement"""

    print("🔍 Alpaca API Debugging Tool (Non-Interactive)")
    print("=" * 50)

    # Try to get credentials from environment variables first
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    # If not found, use the example credentials provided by user for testing
    if not api_key or not secret_key:
        print("⚠️ Environment variables not found, using provided test credentials")
        api_key = '***REMOVED***'
        secret_key = '***REMOVED***'

    print("✅ Found API credentials in environment variables")
    print(f"🔑 API Key: {api_key[:8]}...")
    print(f"🔑 Secret Key: {'✅ Present' if secret_key else '❌ Missing'}")

    print("\n📊 Testing Paper Trading Account...")

    try:
        # Create broker
        broker = AlpacaBroker(api_key, secret_key, paper_trading=True)

        # Test authentication
        print("\n🔐 Testing Authentication:")
        auth_success = broker.test_authentication()

        if not auth_success:
            print("❌ Authentication failed. Check your API credentials.")
            return False

        # Check crypto permissions
        print("\n₿ Checking Crypto Permissions:")
        crypto_ok = broker.check_crypto_permissions()

        if not crypto_ok:
            print("⚠️ Crypto permissions check failed")

        print("\n📊 Checking Current Positions:")
        positions = broker.get_positions()
        if positions:
            print(f"Current positions: {len(positions)}")
            for pos in positions:
                print(f"  {pos.get('symbol', 'Unknown')}: {pos.get('qty', 0)} @ ${pos.get('avg_entry_price', 0)}")
        else:
            print("No current positions")

        print("\n📋 Checking Recent Orders:")
        orders = broker.get_orders("all")
        if orders:
            print(f"Recent orders: {len(orders)}")
            for order in orders[:3]:  # Show last 3 orders
                print(f"  {order.get('symbol', 'Unknown')}: {order.get('side', 'Unknown')} {order.get('qty', 0)} - {order.get('status', 'Unknown')}")
        else:
            print("No recent orders")

        print("\n🧪 Testing Small Crypto Order:")
        print("This will attempt to place a small test order...")

        # Test a small but valid order (1 unit of BTCUSD, which is valid for crypto)
        result = broker.buy("BTCUSD", 1, "market")  # 1 unit of crypto

        if result:
            print("✅ Order placement successful!")
            print(f"📋 Order details: {result}")
            return True
        else:
            print("❌ Order placement failed!")
            return False

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_alpaca_authentication()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")