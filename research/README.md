# Research Tools

This folder contains research tools for backtesting and optimizing trading strategies.

## Contents

### Core Files

**Backtest Implementation:**
- **`Backtest.java`** - Java implementation of mean reversion backtesting
- **`Backtest.class`** - Compiled Java bytecode
- **`Backtest$*.class`** - Inner class bytecode files

**Utilities:**
- **`fetch_polygon_data.py`** - Fetch historical data from Polygon API
- **`find_best.py`** - Parameter optimization with walk-forward analysis
- **`build.sh`** - Build script to compile the backtest
- **`btc_data.csv`** - Sample Bitcoin data

## Quick Start

### 1. Fetch Data

```bash
# Fetch Bitcoin minute data (last 7 days)
python fetch_polygon_data.py X:BTCUSD minute 7 btc_data.csv YOUR_API_KEY

# Fetch stock daily data (last year)
python fetch_polygon_data.py AAPL day 365 aapl_data.csv YOUR_API_KEY

# Or set API key as environment variable
export POLYGON_API_KEY=your_key_here
python fetch_polygon_data.py SPY hour 30 spy_data.csv
```

### 2. Compile and Run Backtest

```bash
# Compile the backtest
javac Backtest.java

# Or use the build script
./build.sh

# Run with default parameters (SMA=20, Std=2.0)
java Backtest btc_data.csv

# Run with custom parameters (SMA=30, Std=2.5)
java Backtest btc_data.csv 30 2.5
```

### 3. Find Optimal Parameters

```bash
# Run walk-forward optimization
python3 find_best.py btc_data.csv
```

The optimizer will:
- Split data into train (50%), validation (25%), test (25%)
- Test **680+ parameter combinations** on training data
  - SMA periods: 1-100 (28 values tested)
  - Std multipliers: 0.1-4.0 (22 values tested)
- Validate top 10 on both validation and test sets
- Recommend best parameters with robustness analysis
- Display progress with ETA estimates

## Mean Reversion Strategy

The strategy uses Bollinger Bands for mean reversion:

**Entry Signals:**
- **Buy**: Price crosses below lower band (SMA - Std × σ)
- **Short**: Price crosses above upper band (SMA + Std × σ)

**Exit Signals:**
- **Exit Long**: Price returns to SMA
- **Exit Short**: Price returns to SMA

**Parameters:**
- `sma_period`: Moving average lookback (default: 20)
- `std_multiplier`: Standard deviation multiplier (default: 2.0)

## Parameter Optimization

### Walk-Forward Analysis

The `find_best.py` script implements proper walk-forward optimization to prevent overfitting:

```
Data Split:
├─ Training Set (50%)    → Optimize parameters
├─ Validation Set (25%)  → First out-of-sample test
└─ Test Set (25%)        → Second out-of-sample test
```

**Parameter Ranges Tested:**
- **SMA Periods:** 28 values from 1 to 100
  - 1-10: Every value (high frequency scalping)
  - 12-20: Every 2 (short-term trading)
  - 25-50: Every 5 (medium-term trading)
  - 60-100: Every 10 (long-term trading)
- **Std Multipliers:** 22 values from 0.1 to 4.0
  - 0.1-1.0: Every 0.1 (tight bands)
  - 1.25-4.0: Every 0.25 (wide bands)
- **Total combinations:** 616 parameter sets

**Selection Criteria:**
1. Positive P&L on training set
2. Consistent performance on validation and test sets
3. Low overfitting (validation P&L > 30% of training P&L)
4. Good generalization (test P&L > 30% of training P&L)

### Example Output

```
RECOMMENDED PARAMETERS
=====================
Best parameters based on out-of-sample performance:
  SMA Period:        20
  Std Multiplier:    2.0

Performance:
  Training P&L:      $1250.50 (45 trades)
  Validation P&L:    $890.25 (23 trades)
  Test P&L:          $765.80 (21 trades)
  Avg Out-of-Sample: $828.03

Robustness Analysis:
  Consistent performers: 4/5 (80%)
  (Positive P&L on both validation and test sets)

Overfitting Check:
  Validation/Training ratio: 0.71
  Test/Training ratio:       0.61
  ✓ Good generalization - parameters perform well on unseen data
```

## CSV Data Format

All tools expect CSV files with the following format:

```csv
timestamp,open,high,low,close,volume
2024-01-01T09:30:00Z,150.25,151.00,150.00,150.75,1000000
2024-01-01T09:31:00Z,150.75,151.50,150.50,151.25,1200000
```

The `fetch_polygon_data.py` script automatically formats data correctly.

## Performance

### Java Backtest
- Handles up to 100,000+ bars efficiently
- Typical runtime: ~150ms for 43,000 bars
- Cross-platform compatibility (Windows, macOS, Linux)
- Automatic memory management (no memory leaks)

### Python Optimizer
- Tests 616 parameter combinations × 3 data splits = 1,848 backtests
- Expected runtime: 3-8 minutes depending on data size
- Progress displayed with ETA estimates
- Results saved to timestamped CSV for later analysis
- Parallel processing potential for future optimization

## Advanced Usage

### Custom Parameter Ranges

Edit `find_best.py` (lines 193-204) to test different parameter ranges:

```python
# Example: Test only high-frequency parameters (faster optimization)
sma_periods = list(range(1, 21))  # 1-20
std_multipliers = [round(x * 0.1, 1) for x in range(1, 31)]  # 0.1-3.0

# Example: Test only long-term parameters
sma_periods = list(range(50, 201, 10))  # 50-200
std_multipliers = [round(x * 0.5, 1) for x in range(2, 9)]  # 1.0-4.0

# Example: Dense grid search around known good values
sma_periods = list(range(15, 36))  # 15-35 (every value)
std_multipliers = [round(1.0 + x * 0.1, 1) for x in range(21)]  # 1.0-3.0
```

### Analyzing Results

```python
import pandas as pd

# Load optimization results
df = pd.read_csv('optimization_results_20250321_143022.csv')

# Find most consistent parameters
consistent = df[df['consistency'] == 1.0]
best = consistent.nlargest(3, 'avg_pnl')
print(best[['sma_period', 'std_multiplier', 'avg_pnl']])

# Compare in-sample vs out-of-sample
df['oos_ratio'] = (df['val_pnl'] + df['test_pnl']) / (2 * df['train_pnl'])
print(df.nlargest(5, 'oos_ratio')[['sma_period', 'std_multiplier', 'oos_ratio']])
```

## Files Generated

- `*_train.csv` - Training data split (temporary)
- `*_validation.csv` - Validation data split (temporary)
- `*_test.csv` - Test data split (temporary)
- `optimization_results_*.csv` - Optimization results with timestamp

Temporary split files are automatically cleaned up after optimization completes.

## Workflow Example

```bash
# 1. Fetch fresh data
python fetch_polygon_data.py X:BTCUSD minute 7 btc_latest.csv

# 2. Find optimal parameters
python find_best.py btc_latest.csv

# Output shows recommended parameters, e.g., SMA=25, Std=2.0

# 3. Run full backtest with optimal parameters
./backtest btc_latest.csv 25 2.0

# 4. Review results and integrate into live trading
```

## Notes

- Always validate on out-of-sample data before live trading
- Market conditions change - re-optimize periodically
- Consider transaction costs in live trading (not included in backtest)
- The C implementation is ~100x faster than Python for large datasets
- Use the optimizer to find parameters, then validate with full backtest

## Requirements

**Java:**
- JDK 8 or higher for compilation
- JRE 8 or higher for execution

**Python Dependencies:**
```bash
pip install pandas numpy requests
```

**API Access:**
- Polygon.io API key (free tier available)

## Verified Performance

Tested on 43,060 Bitcoin minute bars with SMA=20, Std=2.0:

```
Trading Statistics:
  Total Trades:      1660
  Winning Trades:    1209
  Losing Trades:     451

Performance Metrics:
  Total P&L:         $70,986.60
  Max Drawdown:      $15,622.57
  Win Rate:          72.83%
  Average Win:       $141.82
  Average Loss:      $222.77
  Profit Factor:     1.71

Execution Time:    ~150ms
```
