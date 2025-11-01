# cython: language_level=3
# distutils: language = c++

import sys
from libc.math cimport sqrt

cdef class Bar:
    """Represents a single OHLCV bar"""
    cdef public str timestamp
    cdef public double open
    cdef public double high
    cdef public double low
    cdef public double close
    cdef public double volume

    def __init__(self, str timestamp, double open, double high, double low, double close, double volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


cdef class TradingState:
    """Tracks the current trading state"""
    cdef public int position
    cdef public double entry_price
    cdef public double total_pnl
    cdef public double peak_equity
    cdef public double max_drawdown
    cdef public int total_trades
    cdef public int winning_trades
    cdef public int losing_trades
    cdef public double total_wins
    cdef public double total_losses

    def __init__(self):
        self.position = 0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.peak_equity = 0.0
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_wins = 0.0
        self.total_losses = 0.0


cpdef list load_csv_data(str filename, bint verbose=True):
    """Load CSV data from file"""
    cdef list bars = []
    cdef str line
    cdef list fields
    cdef str timestamp
    cdef double open_val, high_val, low_val, close_val, volume_val
    cdef bint header_skipped = False

    try:
        with open(filename, 'r') as f:
            for line in f:
                # Skip header line
                if not header_skipped:
                    header_skipped = True
                    continue

                # Parse CSV line: timestamp,open,high,low,close,volume
                fields = line.strip().split(',')

                if len(fields) >= 5:
                    try:
                        timestamp = fields[5].strip()
                        open_val = float(fields[1].strip())
                        high_val = float(fields[3].strip())
                        low_val = float(fields[4].strip())
                        close_val = float(fields[2].strip())
                        volume_val = float(fields[0].strip()) if len(fields) > 5 else 0.0

                        bars.append(Bar(timestamp, open_val, high_val, low_val, close_val, volume_val))
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        if verbose:
                            print(f"Warning: Skipping malformed line: {line.strip()}", file=sys.stderr)

        if verbose:
            print(f"Loaded {len(bars)} bars from {filename}")
        return bars

    except FileNotFoundError:
        if verbose:
            print(f"Error: File not found: {filename}", file=sys.stderr)
        return []
    except IOError as e:
        if verbose:
            print(f"Error reading file: {e}", file=sys.stderr)
        return []


cpdef double calculate_sma(list bars, int current_idx, int period):
    """Calculate Simple Moving Average"""
    cdef double sum_val = 0.0
    cdef int i
    cdef Bar bar

    if current_idx < period - 1:
        return 0.0

    for i in range(period):
        bar = bars[current_idx - i]
        sum_val += bar.close

    return sum_val / period


cpdef double calculate_std(list bars, int current_idx, int period, double mean):
    """Calculate Standard Deviation"""
    cdef double sum_sq_diff = 0.0
    cdef double diff
    cdef int i
    cdef Bar bar

    if current_idx < period - 1:
        return 0.0

    for i in range(period):
        bar = bars[current_idx - i]
        diff = bar.close - mean
        sum_sq_diff += diff * diff

    return sqrt(sum_sq_diff / period)


cpdef void execute_strategy(list bars, TradingState state, int sma_period, double std_multiplier, bint verbose=True):
    """Execute mean reversion strategy"""
    cdef int i
    cdef double sma, std, upper_band, lower_band, current_price, pnl, current_drawdown
    cdef str timestamp
    cdef Bar bar

    for i in range(sma_period, len(bars)):
        sma = calculate_sma(bars, i, sma_period)
        std = calculate_std(bars, i, sma_period, sma)

        if sma == 0.0 or std == 0.0:
            continue

        upper_band = sma + (std_multiplier * std)
        lower_band = sma - (std_multiplier * std)
        bar = bars[i]
        current_price = bar.close
        timestamp = bar.timestamp

        # Entry signals
        if state.position == 0:
            # Buy signal: price crosses below lower band
            if current_price < lower_band:
                state.position = 1
                state.entry_price = current_price
                if verbose:
                    print(f"BUY at {timestamp}: Price={current_price:.2f}, SMA={sma:.2f}, Lower Band={lower_band:.2f}")
            # Short signal: price crosses above upper band
            elif current_price > upper_band:
                state.position = -1
                state.entry_price = current_price
                if verbose:
                    print(f"SHORT at {timestamp}: Price={current_price:.2f}, SMA={sma:.2f}, Upper Band={upper_band:.2f}")

        # Exit signals
        elif state.position == 1:
            # Exit long when price returns to mean
            if current_price >= sma:
                pnl = current_price - state.entry_price
                state.total_pnl += pnl
                state.total_trades += 1

                if pnl > 0:
                    state.winning_trades += 1
                    state.total_wins += pnl
                else:
                    state.losing_trades += 1
                    state.total_losses += abs(pnl)

                if verbose:
                    print(f"SELL at {timestamp}: Price={current_price:.2f}, Entry={state.entry_price:.2f}, PnL={pnl:.2f}")
                state.position = 0

        elif state.position == -1:
            # Exit short when price returns to mean
            if current_price <= sma:
                pnl = state.entry_price - current_price
                state.total_pnl += pnl
                state.total_trades += 1

                if pnl > 0:
                    state.winning_trades += 1
                    state.total_wins += pnl
                else:
                    state.losing_trades += 1
                    state.total_losses += abs(pnl)

                if verbose:
                    print(f"COVER at {timestamp}: Price={current_price:.2f}, Entry={state.entry_price:.2f}, PnL={pnl:.2f}")
                state.position = 0

        # Track drawdown
        if state.total_pnl > state.peak_equity:
            state.peak_equity = state.total_pnl

        current_drawdown = state.peak_equity - state.total_pnl
        if current_drawdown > state.max_drawdown:
            state.max_drawdown = current_drawdown


cpdef void print_results(TradingState state):
    """Print backtest results"""
    print()
    print("========================================")
    print("       BACKTEST RESULTS ANALYSIS        ")
    print("========================================")
    print()

    print("Trading Statistics:")
    print(f"  Total Trades:      {state.total_trades}")
    print(f"  Winning Trades:    {state.winning_trades}")
    print(f"  Losing Trades:     {state.losing_trades}")
    print()

    print("Performance Metrics:")
    print(f"  Total P&L:         ${state.total_pnl:.2f}")
    print(f"  Max Drawdown:      ${state.max_drawdown:.2f}")

    if state.total_trades > 0:
        win_rate = (state.winning_trades / state.total_trades) * 100
        print(f"  Win Rate:          {win_rate:.2f}%")

        if state.winning_trades > 0:
            avg_win = state.total_wins / state.winning_trades
            print(f"  Average Win:       ${avg_win:.2f}")
        else:
            print("  Average Win:       $0.00")

        if state.losing_trades > 0:
            avg_loss = state.total_losses / state.losing_trades
            print(f"  Average Loss:      ${avg_loss:.2f}")
        else:
            print("  Average Loss:      $0.00")

        if state.winning_trades > 0 and state.losing_trades > 0:
            profit_factor = state.total_wins / state.total_losses
            print(f"  Profit Factor:     {profit_factor:.2f}")
        elif state.winning_trades > 0 and state.losing_trades == 0:
            print("  Profit Factor:     ∞ (no losses)")
        else:
            print("  Profit Factor:     0.00")

        # Calculate Expectancy: (WinRate × AverageWin) - (LossRate × AverageLoss)
        avg_win = state.total_wins / state.winning_trades if state.winning_trades > 0 else 0.0
        avg_loss = state.total_losses / state.losing_trades if state.losing_trades > 0 else 0.0
        loss_rate = (state.losing_trades / state.total_trades) * 100
        expectancy = (win_rate / 100.0 * avg_win) - (loss_rate / 100.0 * avg_loss)
        print(f"  Expectancy:        ${expectancy:.2f}")

    print()
    print("========================================")


def run_backtest_silent(str filename, int sma_period, double std_multiplier):
    """
    Run backtest silently and return metrics as a dictionary

    Args:
        filename: Path to CSV file with OHLCV data
        sma_period: Period for Simple Moving Average
        std_multiplier: Standard deviation multiplier for bands

    Returns:
        Dictionary with backtest metrics, or None if error
    """
    # Load data silently
    bars = load_csv_data(filename, verbose=False)

    if not bars:
        return None

    # Initialize trading state
    state = TradingState()

    # Run backtest silently
    execute_strategy(bars, state, sma_period, std_multiplier, verbose=False)

    # Calculate metrics
    cdef dict metrics = {
        'sma_period': sma_period,
        'std_multiplier': std_multiplier,
        'total_trades': state.total_trades,
        'winning_trades': state.winning_trades,
        'losing_trades': state.losing_trades,
        'total_pnl': state.total_pnl,
        'max_drawdown': state.max_drawdown,
        'win_rate': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'profit_factor': 0.0,
        'expectancy': 0.0
    }

    # Calculate derived metrics
    if state.total_trades > 0:
        metrics['win_rate'] = (state.winning_trades / state.total_trades) * 100.0

        if state.winning_trades > 0:
            metrics['avg_win'] = state.total_wins / state.winning_trades

        if state.losing_trades > 0:
            metrics['avg_loss'] = state.total_losses / state.losing_trades

        if state.winning_trades > 0 and state.losing_trades > 0:
            metrics['profit_factor'] = state.total_wins / state.total_losses
        elif state.winning_trades > 0 and state.losing_trades == 0:
            metrics['profit_factor'] = 999.99  # Very high value to indicate perfect strategy

        # Calculate Expectancy: (WinRate × AverageWin) - (LossRate × AverageLoss)
        avg_win = metrics['avg_win']
        avg_loss = metrics['avg_loss']
        loss_rate = (state.losing_trades / state.total_trades) * 100.0
        metrics['expectancy'] = (metrics['win_rate'] / 100.0 * avg_win) - (loss_rate / 100.0 * avg_loss)

    return metrics


def run_backtest(str filename, int sma_period=20, double std_multiplier=2.0):
    """
    Main backtest function

    Args:
        filename: Path to CSV file with OHLCV data
        sma_period: Period for Simple Moving Average (default: 20)
        std_multiplier: Standard deviation multiplier for bands (default: 2.0)
    """
    print("========================================")
    print("   MEAN REVERSION BACKTEST")
    print("========================================")
    print(f"Data File:         {filename}")
    print(f"SMA Period:        {sma_period}")
    print(f"Std Multiplier:    {std_multiplier:.1f}")
    print("========================================")
    print()

    # Load data
    bars = load_csv_data(filename)

    if not bars:
        print("Error: No data loaded from file", file=sys.stderr)
        sys.exit(1)

    # Initialize trading state
    state = TradingState()

    # Run backtest
    print("Running backtest...")
    print()
    execute_strategy(bars, state, sma_period, std_multiplier)

    # Print results
    print_results(state)


def main():
    """Command-line entry point"""
    if len(sys.argv) < 2:
        print("Usage: python backtest_run.py <csv_file> [sma_period] [std_multiplier]")
        print("Example: python backtest_run.py data.csv 20 2.0")
        sys.exit(1)

    filename = sys.argv[1]
    sma_period = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    std_multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

    run_backtest(filename, sma_period, std_multiplier)


if __name__ == "__main__":
    main()
