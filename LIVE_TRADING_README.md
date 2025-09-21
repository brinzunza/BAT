# BTC/USD Live Trading System with Alpaca

This system integrates live data fetching from Alpaca with real-time charting and automated trading for BTC/USD using various technical analysis strategies.

## Features

- 📊 **Live Data Fetching**: Real-time BTC/USD data from Alpaca Markets
- 📈 **Interactive Charts**: Live candlestick charts with technical indicators
- 🤖 **Automated Trading**: Execute trades based on strategy signals
- 🛡️ **Paper Trading**: Safe testing with simulated trades
- 📋 **Multiple Strategies**: Bollinger Bands, RSI, MACD strategies
- 📊 **Performance Tracking**: Real-time P&L and trade statistics

## Files Created/Modified

### New Files:
- `data_providers/alpaca_provider.py` - Alpaca API integration for data and trading
- `live_trading_chart.py` - Live charting with strategy indicators and trading
- `run_live_trading.py` - Main script to run the live trading system
- `demo_live_trading.py` - Demo script with simulated data

### Key Components:

1. **AlpacaDataProvider**: Fetches live crypto data from Alpaca
2. **AlpacaBroker**: Handles trade execution through Alpaca API
3. **LiveTradingChart**: Real-time chart with strategy indicators
4. **LiveTradingEngine**: Manages trading logic and position tracking

## Quick Start

### 1. Demo Mode (No API Required)
```bash
python3 demo_live_trading.py
```
This runs a demo with simulated BTC data to show how the system works.

### 2. Live Trading with Alpaca

#### Setup Alpaca Account
1. Create a free account at [Alpaca Markets](https://alpaca.markets/)
2. Get your API keys from the dashboard
3. Set environment variables:
```bash
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
```

#### Run Live Trading
```bash
python3 run_live_trading.py
```

The script will:
- Prompt for API credentials if not set in environment
- Let you choose a trading strategy
- Start live trading with real-time charts
- Update every 60 seconds
- Execute paper trades automatically

## Available Strategies

1. **Bollinger Bands**: Buy when price touches lower band, sell when touching upper band
2. **RSI Strategy**: Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)
3. **MACD Strategy**: Buy/sell based on MACD line crossovers

## Configuration

### Default Settings:
- **Symbol**: BTC/USD
- **Paper Trading**: Enabled (for safety)
- **Initial Balance**: $10,000
- **Position Size**: 0.01 BTC
- **Update Interval**: 60 seconds

### Customization:
You can modify these settings in `run_live_trading.py`:
```python
symbol = "BTC/USD"
paper_trading = True
initial_balance = 10000
quantity = 0.01
```

## Chart Features

The live chart displays:
- **Candlestick Chart**: Real-time BTC/USD price action
- **Technical Indicators**: Strategy-specific indicators (BB, RSI, MACD)
- **Trading Signals**: Buy/sell signals marked on chart
- **Performance Stats**: Real-time P&L, position, win rate
- **Position Status**: Current position (LONG/SHORT/FLAT)

## Safety Features

- **Paper Trading Default**: All trades are simulated by default
- **Trade Cooldown**: 5-minute cooldown between trades to prevent over-trading
- **Error Handling**: Robust error handling for API issues
- **Position Limits**: Single position trading (no pyramiding)

## Example Output

```
🚀 BTC/USD Live Trading System
========================================

⚙️  Configuration:
Symbol: BTC/USD
Strategy: Bollinger Bands
Paper Trading: True
Initial Balance: $10,000.00
Position Size: 0.01 BTC
API Status: ✅ Connected to Alpaca

🎯 Starting live trading...
📈 Chart will update every 60 seconds
🔄 Trading signals will be processed automatically
⏹️  Press Ctrl+C to stop
```

## Required Dependencies

The system uses existing dependencies from your project:
- `matplotlib` - For charting
- `pandas` - Data manipulation
- `numpy` - Numerical calculations
- `requests` - API calls
- `datetime` - Time handling

## API Rate Limits

Alpaca has generous rate limits for market data:
- 200 requests per minute for market data
- The system fetches data every 60 seconds, well within limits

## Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure you're running from the project root directory
2. **API Errors**: Check your Alpaca API credentials
3. **No Data**: Ensure markets are open (crypto trades 24/7)
4. **Chart Not Updating**: Check internet connection and API limits

### Debug Mode:
Add debug prints by modifying the `fetch_and_process_data()` method in `live_trading_chart.py`.

## Future Enhancements

Potential improvements:
- WebSocket streaming for faster updates
- Multiple timeframe analysis
- Risk management features
- Email/SMS alerts for trades
- Database logging of trades
- Portfolio optimization

## Disclaimer

This system is for educational and testing purposes. Always:
- Use paper trading first
- Understand the risks of automated trading
- Monitor your positions actively
- Never risk more than you can afford to lose
- Consider market conditions and volatility

## Support

For issues with the trading system, check:
1. Your Alpaca API credentials
2. Network connectivity
3. Market hours (crypto is 24/7)
4. Console output for error messages