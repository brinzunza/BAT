<div style="margin-bottom: 20px;">
  <img src="bdavid.png" alt="BAT Trading System" width="100%" style="display: block;"/>
</div>

# BAT [Bruninvestor Algorithmic Trading]

BAT is a comprehensive algorithmic trading system that supports both **stocks and cryptocurrencies** with multiple trading strategies and execution modes.

## Disclaimer
This project is for educational purposes only. It is not financial advice. Always conduct thorough research or consult with a professional financial advisor before making financial or investment decisions.

## Trading Modes

### 1. Backtesting Mode
- Test strategies on historical data
- Custom time periods
- Risk-free strategy validation
- Interactive performance charts
- Detailed trade analysis

### 2. Live Trading Mode
- Real-time trading execution
- Live candlestick charts
- Automated signal processing
- Real-time P&L tracking

### 3. Research and Optimization
- Data collection for different tickers
- Parameter tuning and testing
- Efficient optimization for paramters

## Features

- **Unified Trading Platform** - Trade both stocks and crypto from one interface
- **Multiple Strategies** - Bollinger Bands, RSI, MACD, Moving Averages
- **Dual Trading Modes** - Backtesting and Live Trading
- **Real-time Charts** - Interactive candlestick charts with indicators
- **Multiple Brokers** - Alpaca Paper Trading and Simulated Broker
- **Comprehensive Analysis** - Detailed performance metrics and visualizations

## Quick Start

### Main Application 
```bash
python3 main.py
```
Choose between backtesting, research, and live trading with custom parameters.
## Monorepo layout & live-bot analytics

`tradingResearch/` (strategy research, `quant_engine`, BEID/DTFX studies; merged here with its full git history) lives alongside the BAT engine.
The old path `~/Documents/GitHub/tradingResearch` is a symlink to `BAT/tradingResearch` so hard-coded absolute paths keep working.

`analytics/prodlive/` queries the live bot (**PRODLIVETRADING**, separate repo/VM) through a read-only snapshot of its SQLite store:
```bash
export PRODLIVE_SSH="user@vm-host:/home/user/PRODLIVETRADING/logs/practice/trades.db"   # or PRODLIVE_DB=/local/path/trades.db
python3 -m analytics.prodlive sync                      # consistent sqlite-backup copy of the bot DB -> analytics/prodlive/data/latest.db
python3 -m analytics.prodlive summary [--symbol EUR_USD --since 2026-09-01 --json]
python3 -m analytics.prodlive by symbol|side|entry_hour_utc|outcome|weekday|month
python3 -m analytics.prodlive weekly | equity | trades | events | open | plot
python3 -m analytics.prodlive expect --win-rate 0.45 --avg-r 0.1     # live vs backtest expectation (z / t)
python3 -m analytics.prodlive sql "select symbol,count(*) from trades group by 1"
```
In Python: `from analytics.prodlive import queries as Q; Q.summary(); Q.breakdown("symbol")`.
