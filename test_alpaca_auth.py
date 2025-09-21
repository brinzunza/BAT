#!/usr/bin/env python3
"""
Test script to diagnose Alpaca API authentication and order placement issues
"""

from data_providers.alpaca_provider import AlpacaBroker

def test_alpaca_authentication():
    """Test Alpaca authentication and order placement"""

    print("🔍 Alpaca API Debugging Tool")
    print("=" * 40)

    # Get credentials from user
    api_key = input("Enter your Alpaca API Key: ").strip()
    secret_key = input("Enter your Alpaca Secret Key: ").strip()

    if not api_key or not secret_key:
        print("❌ API credentials are required")
        return

    print("\n📊 Testing Paper Trading Account...")

    try:
        # Create broker
        broker = AlpacaBroker(api_key, secret_key, paper_trading=True)

        # Test authentication
        print("\n🔐 Testing Authentication:")
        auth_success = broker.test_authentication()

        if not auth_success:
            print("❌ Authentication failed. Check your API credentials.")
            return

        print("\n🧪 Testing Small Crypto Order:")
        print("This will attempt to place a very small test order...")

        # Test a very small order
        result = broker.buy("BTCUSD", 0.001, "market")  # $50-100 order

        if result:
            print("✅ Order placement successful!")
            print(f"📋 Order details: {result}")
        else:
            print("❌ Order placement failed!")

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

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_alpaca_authentication()