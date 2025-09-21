#!/usr/bin/env python3
"""
Demo script showing what the console output looks like during live trading
"""

from datetime import datetime
import time

def demo_console_output():
    """Simulate console output during live trading"""

    print("🚀 BTC/USD Live Trading System - Console Output Demo")
    print("=" * 55)
    print("This shows what you'll see in the console during live trading")
    print("=" * 55)

    # Simulate initial setup
    print("\n⚙️  Configuration Summary:")
    print("   Strategy: Bollinger Bands")
    print("   Symbol: BTC/USD")
    print("   Position Size: 0.01 BTC")
    print("   Update Interval: 60 seconds")
    print("   Trading Mode: Paper")
    print("   Account Balance: Will be retrieved from Alpaca")

    print("\n💰 Account Balance: $10,000.00")  # Retrieved from Alpaca

    print("\n📈 Features:")
    print("   ✅ Real-time candlestick chart")
    print("   ✅ Strategy indicators overlay")
    print("   ✅ Buy/sell signals on chart")
    print("   ✅ Live P&L tracking")
    print("   ✅ Automated trade execution")
    print("   ✅ Console trade logging")

    print("\n🎯 Starting live trading with charts...")
    print("📊 Chart will open in a new window")
    print("🔄 Data updates every 60 seconds")
    print("💡 All trades will be logged to console")
    print("⏹️  Press Ctrl+C to stop")
    print("\n" + "=" * 55)

    # Show fast data collection phase
    print("\n🔄 Starting fast data collection for BTC/USD...")
    print("📊 Fetched 60 recent bars using public endpoint")
    print("✅ Fast initialization complete! Loaded 50 bars")
    print("🎯 Ready for trading immediately!")

    # Simulate market updates
    prices = [45250.30, 45180.75, 45120.50, 44890.25, 44750.80, 45050.60]
    signals = [False, False, True, False, False, True]  # Buy signal at index 2, sell at 5

    for i, (price, signal) in enumerate(zip(prices, signals)):
        time.sleep(1)  # Pause for demo effect

        timestamp = datetime.now()

        print(f"\n📊 [{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Market Update:")
        print(f"   💰 BTC/USD: ${price:.2f}")

        if i == 2:  # Buy signal
            print(f"   📈 Buy Signal: 🟢 YES")
            print(f"   📉 Sell Signal: ⚫ NO")

            print(f"\n🚨 [{timestamp.strftime('%H:%M:%S')}] BUY SIGNAL TRIGGERED!")
            print(f"   💵 Price: ${price:.2f}")
            print(f"   📊 Strategy: Bollinger Bands")
            print(f"   🔄 Processing trade...")
            print(f"   ✅ Trade OPENED!")
            print(f"   📍 New Position: LONG @ ${price:.2f}")
            print(f"   🏦 New Balance: $10,000.00")

        elif i == 5:  # Sell signal
            print(f"   📈 Buy Signal: ⚫ NO")
            print(f"   📉 Sell Signal: 🔴 YES")

            profit = (price - 44890.25) * 0.01  # Calculate profit

            print(f"\n🚨 [{timestamp.strftime('%H:%M:%S')}] SELL SIGNAL TRIGGERED!")
            print(f"   💵 Price: ${price:.2f}")
            print(f"   📊 Strategy: Bollinger Bands")
            print(f"   🔄 Processing trade...")
            print(f"   ✅ Trade CLOSED/CHANGED!")
            print(f"   📍 New Position: FLAT (No position)")
            print(f"   💰 Balance Change: +${profit:.2f}")
            print(f"   🏦 New Balance: ${10000 + profit:.2f}")

        else:  # No signal
            print(f"   📈 Buy Signal: ⚫ NO")
            print(f"   📉 Sell Signal: ⚫ NO")

            if i > 2 and i < 5:  # We have a position
                unrealized = (price - 44890.25) * 0.01
                sign = "+" if unrealized > 0 else ""
                print(f"   📊 Position: LONG @ $44,890.25")
                print(f"   💹 Unrealized P&L: {sign}${unrealized:.2f}")
            else:
                print(f"   📊 Position: FLAT (No position)")

    print(f"\n⏹️  Live trading stopped by user")

    # Final summary
    print(f"\n" + "=" * 40)
    print(f"         FINAL TRADING SUMMARY")
    print(f"=" * 40)
    print(f"Strategy: Bollinger Bands")
    print(f"Total Trades: 2")
    print(f"Profitable Trades: 1")
    print(f"Losing Trades: 0")
    print(f"Win Rate: 100.0%")
    print(f"Final Balance: $10,001.60")
    print(f"Total Return: $1.60")
    print(f"Percent Return: 0.02%")
    print(f"Current Position: 0")

    print(f"\n📈 Recent Trades:")
    print("timestamp                action     price     profit   balance")
    print("2025-01-21 14:30:15     buy_long   44890.25   0.00    10000.00")
    print("2025-01-21 14:35:15     sell_long  45050.60   1.60    10001.60")

    print(f"\n✅ Demo complete!")
    print(f"This is what you'll see when running: python3 main.py -> option 2")

if __name__ == "__main__":
    demo_console_output()