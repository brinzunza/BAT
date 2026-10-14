# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BAT (Bruninvestor Algorithmic Trading) is a comprehensive Python-based algorithmic trading system supporting **backtesting** and **live trading** for stocks, cryptocurrencies, and forex. The system uses a modular architecture with interchangeable strategies, data providers, and broker interfaces.

## Running the System

### Main Application
```bash
python3 main.py
```
This launches the CLI menu with options for backtesting, live trading, and research/optimization.

### Environment Setup
Create a `.env` file in the root directory with:
```bash
POLYGON_API_KEY=your_polygon_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
OANDA_ACCESS_TOKEN=your_oanda_token
OANDA_ACCOUNT_ID=your_oanda_account_id
SYNTH_BASE_URL=http://35.209.219.174:8000
SYNTH_API_KEY=bruno
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Core Components

**Trading Engines** (`src/execution/`):
- `BacktestEngine`: Simulates strategy performance on historical data
- `LiveTradingEngine`: Executes strategies in real-time with broker integration
- Both support long-only and long/short trading modes

**Strategies** (`src/strategies/`):
All strategies implement a common interface:
- `generate_signals(df)`: Returns DataFrame with 'Buy Signal' and 'Sell Signal' columns
- `get_signal_names()`: Returns dict of signal column names
- `get_indicators()`: Returns list of indicator column names for plotting
- `get_required_lookback()`: Returns minimum bars needed for reliable signals
- `validate_data(df)`: Validates data sufficiency

**Data Providers** (`src/data/providers/`):
All inherit from `BaseDataProvider`:
- `PolygonDataProvider`: Stocks (AAPL), crypto (X:BTCUSD), forex (C:EURUSD)
- `AlpacaDataProvider`: Stocks and crypto
- `OandaProvider`: Forex pairs
- `SynthDataProvider`: Testing with synthetic data

**Broker Interfaces** (`src/execution/brokers.py`, `src/execution/ib_broker.py`):
- `SimulatedBroker`: In-memory simulation for backtesting
- `AlpacaBroker`: Paper trading with Alpaca API
- `IBBroker`: Interactive Brokers integration for forex

### Data Flow

**Backtesting:**
1. User selects strategy + parameters via CLI
2. Data provider fetches historical OHLCV data
3. Strategy generates signals (adds Buy/Sell columns to DataFrame)
4. BacktestEngine simulates trades bar-by-bar
5. Results analyzed: win rate, Sharpe ratio, max drawdown, etc.

**Live Trading:**
1. LiveTradingEngine initializes with data provider + broker
2. Main loop (configurable sleep interval, default 60s):
   - Fetch latest candles
   - Generate signals
   - Validate signals (data sufficiency, no conflicts)
   - Get broker position/account status
   - Execute trades if signals present
   - Synchronize internal state with broker (broker is source of truth)
3. Optional real-time chart visualization via `LiveTradingChart`

## Key Design Patterns

**Strategy Pattern**: Strategies are interchangeable; adding a new one doesn't require modifying engine code.

**Abstract Factory**: Data providers implement common interface; switching providers is transparent to strategies.

**Position Synchronization**: The broker API is the source of truth. LiveTradingEngine syncs internal state every iteration to handle manual trades or errors gracefully.

**Realized vs Unrealized P&L**: During backtesting, account worth uses realized-only gains to avoid misleading equity curves. Unrealized P&L is only included in final analysis.

## Critical Implementation Details

### Forex Spread Modeling
Forex trades include bid-ask spread costs:
```python
pip_value = 0.0001 if 'JPY' not in symbol else 0.01
spread_cost = spread_pips * pip_value
buy_price = price + spread_cost  # Buy at ask
```

### Signal Validation
Signals are validated before execution:
- Sufficient lookback data (`len(df) >= strategy.get_required_lookback()`)
- No conflicting signals (buy and sell simultaneously)
- Strategy-specific validation via `validate_signal_conditions()` if implemented

### Long/Short vs Long-Only
- **Long-only**: Buy signals open positions, sell signals close positions
- **Long/short**: Buy/sell signals can reverse positions (close opposite + open new)

### Order Management
- **Market orders**: Immediate execution (default)
- **Limit orders**: Execute at specific price, with 1-minute timeout in long-only mode
- LiveTradingEngine tracks pending orders and cancels expired ones

## Research & Optimization

### Parameter Optimization
```bash
# From research/optimization/
python3 find_best.py
```
Uses grid search over strategy parameters. Backtests on training period and validates on test period to avoid overfitting.

### Dataset Management
Download historical data for reuse:
```python
# Via CLI option 3 (Research/Optimization)
# Saves to research/datasets/
```

## Testing

The system uses vectorized pandas operations for performance:
```python
# Correct (vectorized, ~100x faster):
df['sma'] = df['Close'].rolling(window).mean()

# Avoid (loops are slow):
for i in range(len(df)):
    df.loc[i, 'sma'] = df['Close'].iloc[i-window:i].mean()
```

## Machine Learning Integration

### Bayesian Regime-Switching Model (`MLearning/`)
Hidden Markov Model for detecting market regime changes (bull ↔ bear):
- Uses Forward-Backward, Viterbi, and Gibbs sampling algorithms
- Operates in log-space for numerical stability
- Still in development; training pipeline incomplete

## Common Patterns

### Adding a New Strategy
1. Create new file in `src/strategies/`
2. Implement required methods: `generate_signals()`, `get_signal_names()`, `get_indicators()`, `get_required_lookback()`, `validate_data()`
3. Add to `TradingCLI.strategies` dict in `src/ui/cli_interface.py`
4. Strategy receives OHLCV DataFrame, returns same DataFrame with signal columns added

### Adding a New Data Provider
1. Inherit from `BaseDataProvider` in `src/data/providers/base_provider.py`
2. Implement `get_data()` for historical data and `get_live_data()` for real-time
3. Return standardized DataFrame: columns=['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']
4. Add provider setup in `src/ui/cli_interface.py`

### Debugging Live Trading
Enable verbose mode in LiveTradingEngine for detailed logs:
- Shows broker interface type
- Logs trade execution details
- Displays position synchronization info

## Asset Class Ticker Formats

- **Stocks**: `AAPL`, `MSFT`, `GOOGL`
- **Crypto**: `X:BTCUSD`, `X:ETHUSD` (Polygon format)
- **Forex**: `C:EURUSD`, `C:GBPUSD` (Polygon format)

## Important Notes

- **Provider Caching**: CLI caches authenticated data provider sessions for performance
- **Account Worth Calculation**: Uses realized-only P&L during backtesting; unrealized included only in final analysis
- **Position Percentage**: Configurable per-trade allocation (1-100% of account)
- **Error Handling**: System never crashes; always provides feedback and exits gracefully
- **Numerical Precision**: Use epsilon comparisons for floats; JPY forex pairs use different pip values (0.01 vs 0.0001)

## File Structure

```
BAT/
├── main.py                         # Entry point
├── ui/
│   ├── cli_interface.py            # Main CLI (~1500 lines)
│   └── live_trading_chart.py       # Real-time visualization
├── engines/
│   ├── backtest_engine.py          # Historical backtesting
│   ├── live_trading_engine.py      # Live execution
│   ├── brokers.py                  # Broker interfaces
│   └── ib_broker.py                # Interactive Brokers
├── strategies/
│   ├── moving_average.py
│   ├── rsi_strategy.py
│   ├── macd_strategy.py
│   ├── bollinger_bands_strategy.py
│   ├── mean_reversion.py
│   └── candlestick_strategy.py
├── src/data/providers/
│   ├── base_provider.py
│   ├── polygon_provider.py
│   ├── alpaca_provider.py
│   ├── oanda_provider.py
│   └── synth_provider.py
├── indicators/
│   └── technical_indicators.py     # SMA, EMA, RSI, MACD, Bollinger Bands
├── MLearning/                      # Bayesian regime-switching (in progress)
└── research/
    ├── datasets/                   # Downloaded historical data
    └── optimization/
        ├── find_best.py            # Parameter optimization
        └── backtest_run.py         # Batch backtesting
```

## Performance Considerations

- Backtest speed: ~8,000 candles/second (vectorized pandas)
- Memory usage: ~200 MB for 1-year of 1-minute data
- Live trading loop: 60-second default interval (configurable)
- For high-frequency strategies, consider WebSocket streaming (future enhancement)
