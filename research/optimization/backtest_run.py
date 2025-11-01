#!/usr/bin/env python3
"""
Mean Reversion Backtest in Cython

COMPILATION:
  python setup.py build_ext --inplace

USAGE:
  python backtest_run.py <csv_file> [sma_period] [std_multiplier]

ARGUMENTS:
  csv_file        - Path to CSV file with OHLCV data (required)
  sma_period      - Period for Simple Moving Average (default: 20)
  std_multiplier  - Standard deviation multiplier for bands (default: 2.0)

EXAMPLES:
  # Basic usage with default parameters (20-period SMA, 2.0 std)
  python backtest_run.py polygon_data.csv

  # Custom parameters: 30-period SMA with 2.5 standard deviations
  python backtest_run.py polygon_data.csv 30 2.5

CSV FORMAT:
  The CSV file must have a header row and the following columns:
  timestamp,open,high,low,close,volume

  Example:
  timestamp,open,high,low,close,volume
  2024-01-01T09:30:00Z,150.25,151.00,150.00,150.75,1000000
  2024-01-01T09:31:00Z,150.75,151.50,150.50,151.25,1200000

STRATEGY:
  Mean Reversion using Bollinger Bands:
  - Buy when price crosses below lower band (mean - std_multiplier * std)
  - Exit long when price returns to mean
  - Short when price crosses above upper band (mean + std_multiplier * std)
  - Exit short when price returns to mean

OUTPUT:
  - Trade log showing all entries and exits
  - Performance metrics including P&L, win rate, profit factor, etc.
"""

import sys

try:
    import backtest
except ImportError:
    print("Error: Cython module not built. Please run:")
    print("  python setup.py build_ext --inplace")
    print()
    print("If you don't have Cython installed, run:")
    print("  pip install cython numpy")
    sys.exit(1)


if __name__ == "__main__":
    backtest.main()
