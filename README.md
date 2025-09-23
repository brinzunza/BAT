<div style="display: flex; align-items: center; margin-bottom: 20px;">
  <img src="bdavid.png" alt="BAT Trading System" width="400" style="margin-right: 20px;"/>
  <h1 style="font-size: 400px; line-height: 400px; margin: 0;">BAT</h1>
</div>

# Bruninvestor Algorithmic Trading

BAT is a comprehensive algorithmic trading system that supports both **stocks and cryptocurrencies** with multiple trading strategies and execution modes.

## 🚀 Key Features

- **Unified Trading Platform** - Trade both stocks and crypto from one interface
- **Multiple Strategies** - Bollinger Bands, RSI, MACD, Moving Averages
- **Dual Trading Modes** - Backtesting and Live Trading
- **Real-time Charts** - Interactive candlestick charts with indicators
- **Multiple Brokers** - Alpaca Paper Trading and Simulated Broker
- **Comprehensive Analysis** - Detailed performance metrics and visualizations

## 📊 Supported Assets

### Cryptocurrencies
- BTC/USD, ETH/USD, DOGE/USD, LTC/USD
- BCH/USD, AVAX/USD, LINK/USD, UNI/USD
- Custom crypto pairs supported

### Stocks
- Major stocks: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX
- ETFs: SPY, QQQ
- Custom stock symbols supported

## 🎯 Quick Start

### Main Application (Recommended)
```bash
python3 main.py
```
Choose between backtesting and live trading with full symbol selection.

### Individual Scripts
```bash
# Backtesting only
python3 run_backtest.py

# Live trading only
python3 run_live_trading.py
```

## 📈 Trading Modes

### 1. Backtesting Mode
- Test strategies on historical data
- Multiple time periods (7 days to 1 year)
- Risk-free strategy validation
- Interactive performance charts
- Detailed trade analysis

### 2. Live Trading Mode
- Real-time trading execution
- Live candlestick charts
- Automated signal processing
- Real-time P&L tracking
- Multiple broker options

## Strategies

### Mean Reversion Strategy
This strategy is based on the assumption that asset prices tend to revert to their historical average over time. 
The algorithm:
- Buys when prices fall below the historical mean*
- Sells when prices rise above the historical mean*

### Moving Averages Strategy
This strategy uses three moving averages (short, medium, and long-term) to identify trends and generate trading signals:
- Buy signals occur when shorter moving averages cross above longer ones*
- Sell signals occur when shorter moving averages cross below longer ones*

## Future Improvements
- Implement additional strategies (e.g., momentum, sentiment analysis)
- Optimize parameters using machine learning techniques
- Integrate real-time data feeds for live trading

## Disclaimer
This project is for educational purposes only. It is not financial advice. Always conduct thorough research or consult with a professional financial advisor before making investment decisions.

*Typical application of strategy