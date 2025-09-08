# BAT - Backtesting & Automated Trading System

A modular trading system that extracts and improves upon the strategies from your Jupyter notebooks, providing both backtesting and live trading capabilities with a clean interface.

## 🚀 Features

- **Modular Strategy System**: Easy-to-extend strategy framework
- **Multiple Data Providers**: Currently supports Polygon.io
- **Backtesting Engine**: Comprehensive backtesting with performance metrics
- **Live Trading**: Paper and live trading support via Alpaca
- **Multiple Strategies**: Mean Reversion, Moving Average, and Trend Structure
- **Clean CLI Interface**: User-friendly command-line interface

## 📁 Project Structure

```
BAT/
├── strategies/           # Trading strategy modules
│   ├── base_strategy.py
│   ├── mean_reversion.py
│   ├── moving_average.py
│   └── trend_structure.py
├── data_providers/       # Data provider modules
│   ├── base_provider.py
│   └── polygon_provider.py
├── engines/             # Backtesting and live trading engines
│   ├── backtest_engine.py
│   ├── live_trading_engine.py
│   └── brokers.py
├── ui/                  # User interface
│   └── cli_interface.py
├── examples/            # Usage examples
│   ├── backtest_example.py
│   └── live_trading_example.py
├── main.py             # Main application entry point
└── requirements.txt    # Python dependencies
```

## 🛠 Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys**:
   - Get a free API key from [Polygon.io](https://polygon.io)
   - For live trading, set up [Alpaca](https://alpaca.markets) account

## 🎯 Usage

### Quick Start

Run the main application:
```bash
python main.py
```

This will launch an interactive CLI where you can:
1. Configure data providers and brokers
2. Select and configure trading strategies  
3. Run backtests or live trading

### Example Usage

#### Backtesting
```python
from strategies.mean_reversion import MeanReversionStrategy
from data_providers.polygon_provider import PolygonDataProvider
from engines.backtest_engine import BacktestEngine

# Setup
data_provider = PolygonDataProvider("your-api-key")
strategy = MeanReversionStrategy(window=20, num_std=2.0)
engine = BacktestEngine(initial_balance=10000)

# Get data and run backtest
df = data_provider.get_data("C:EURUSD", "minute", "2023-01-01", "2023-02-01")
results = engine.backtest(df, strategy)
engine.print_analysis(results)
```

#### Live Trading
```python
from engines.live_trading_engine import LiveTradingEngine
from engines.brokers import SimulatedBroker

# Setup
broker = SimulatedBroker(10000)
engine = LiveTradingEngine(data_provider, broker)

# Run strategy (simulated)
engine.run_strategy(strategy, "BTC", quantity=0.1, sleep_interval=60)
```

## 📊 Available Strategies

### 1. Mean Reversion Strategy
Based on Bollinger Bands - buys when price is below lower band, sells when above upper band.

**Parameters:**
- `window`: Moving average period (default: 20)
- `num_std`: Standard deviations for bands (default: 2.0)

### 2. Moving Average Strategy  
Triple moving average crossover system using short, medium, and long-term averages.

**Parameters:**
- `short_window`: Short MA period (default: 1)
- `medium_window`: Medium MA period (default: 5) 
- `long_window`: Long MA period (default: 25)

### 3. Trend Structure Strategy
Market structure analysis based on swing highs and lows (fixed version of your dtfx notebook).

## 🔧 Configuration

### Data Provider Setup
```python
from data_providers.polygon_provider import PolygonDataProvider
provider = PolygonDataProvider("your-polygon-api-key")
```

### Broker Setup
```python
# Simulated trading
from engines.brokers import SimulatedBroker
broker = SimulatedBroker(initial_balance=10000)

# Live trading with Alpaca
from engines.brokers import AlpacaBroker
broker = AlpacaBroker("api-key", "secret-key", "https://paper-api.alpaca.markets/")
```

## 📈 Performance Metrics

The system tracks:
- Win rate percentage
- Total return and percentage return
- Average profit per trade
- Largest win/loss
- Number of trades
- Balance progression over time

## 🛡 Risk Management

- Position sizing controls
- Simulated trading environment for testing
- Paper trading support via Alpaca
- Comprehensive logging for live trading

## 🔄 Extending the System

### Adding New Strategies
1. Inherit from `BaseStrategy`
2. Implement `generate_signals()` and `get_signal_names()`
3. Add to the strategy selection in CLI

### Adding New Data Providers  
1. Inherit from `BaseDataProvider`
2. Implement `get_data()` and `get_live_data()`
3. Follow the standard DataFrame format

### Adding New Brokers
1. Inherit from `BrokerInterface` 
2. Implement `buy()`, `sell()`, and `get_account_info()`

## 📝 Notes

- This system modularizes the code from your existing notebooks while preserving the original logic
- Your original notebook files remain unchanged
- All strategies have been tested and improved from the notebook versions
- The trend structure strategy has been fixed from the original dtfx.ipynb version

## ⚠️ Disclaimer

This software is for educational and research purposes. Always test strategies thoroughly before using real money. Past performance does not guarantee future results.