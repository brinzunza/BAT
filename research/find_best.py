#!/usr/bin/env python3
"""
Walk-Forward Optimization for Mean Reversion Strategy

This script finds the optimal parameters for the mean reversion strategy by:
1. Training on the first 50% of data (optimization set)
2. Testing on the 3rd quarter (validation set)
3. Testing on the 4th quarter (out-of-sample test set)

This approach prevents overfitting by ensuring the best parameters are validated
on truly unseen data.

Usage:
    python find_best.py <csv_file>

Example:
    python find_best.py btc_data.csv
"""

import sys
import subprocess
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import csv


def split_data(csv_file: str) -> Tuple[str, str, str]:
    """
    Split data into training (50%), validation (25%), and test (25%) sets

    Args:
        csv_file: Path to the input CSV file

    Returns:
        Tuple of (train_file, validation_file, test_file) paths
    """
    print(f"\n{'='*60}")
    print("DATA SPLITTING")
    print(f"{'='*60}")
    print(f"Reading data from: {csv_file}")

    # Read the CSV
    df = pd.read_csv(csv_file)
    total_rows = len(df)

    print(f"Total bars: {total_rows}")

    # Calculate split points
    train_end = int(total_rows * 0.5)
    validation_end = int(total_rows * 0.75)

    # Split the data
    train_df = df.iloc[:train_end]
    validation_df = df.iloc[train_end:validation_end]
    test_df = df.iloc[validation_end:]

    print(f"\nData splits:")
    print(f"  Training set:   {len(train_df):6d} bars (50%)")
    print(f"  Validation set: {len(validation_df):6d} bars (25%)")
    print(f"  Test set:       {len(test_df):6d} bars (25%)")

    # Create temporary files
    base_name = os.path.splitext(csv_file)[0]
    train_file = f"{base_name}_train.csv"
    validation_file = f"{base_name}_validation.csv"
    test_file = f"{base_name}_test.csv"

    # Save splits
    train_df.to_csv(train_file, index=False)
    validation_df.to_csv(validation_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"\nTemporary files created:")
    print(f"  {train_file}")
    print(f"  {validation_file}")
    print(f"  {test_file}")

    return train_file, validation_file, test_file


def run_backtest(csv_file: str, sma_period: int, std_multiplier: float) -> Dict:
    """
    Run the Java backtest with given parameters and parse results

    Args:
        csv_file: Path to CSV data file
        sma_period: SMA period parameter
        std_multiplier: Standard deviation multiplier

    Returns:
        Dictionary with backtest results
    """
    script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else '.'

    # Check if Java backtest is compiled
    java_class = os.path.join(script_dir, 'Backtest.class')

    if not os.path.exists(java_class):
        print(f"Error: Backtest.class not found")
        print("Please compile first: javac Backtest.java")
        print("Or run: ./build.sh")
        sys.exit(1)

    # Run Java backtest
    cmd = ['java', '-cp', script_dir, 'Backtest', csv_file, str(sma_period), str(std_multiplier)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout

        # Parse the output to extract metrics
        metrics = {
            'sma_period': sma_period,
            'std_multiplier': std_multiplier,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0
        }

        # Parse output line by line
        for line in output.split('\n'):
            line = line.strip()

            if 'Total Trades:' in line:
                metrics['total_trades'] = int(line.split(':')[1].strip())
            elif 'Winning Trades:' in line:
                metrics['winning_trades'] = int(line.split(':')[1].strip())
            elif 'Losing Trades:' in line:
                metrics['losing_trades'] = int(line.split(':')[1].strip())
            elif 'Total P&L:' in line:
                pnl_str = line.split(':')[1].strip().replace('$', '').replace(',', '')
                metrics['total_pnl'] = float(pnl_str)
            elif 'Max Drawdown:' in line:
                dd_str = line.split(':')[1].strip().replace('$', '').replace(',', '')
                metrics['max_drawdown'] = float(dd_str)
            elif 'Win Rate:' in line:
                wr_str = line.split(':')[1].strip().replace('%', '')
                metrics['win_rate'] = float(wr_str)
            elif 'Average Win:' in line:
                avg_win_str = line.split(':')[1].strip().replace('$', '').replace(',', '')
                metrics['avg_win'] = float(avg_win_str)
            elif 'Average Loss:' in line:
                avg_loss_str = line.split(':')[1].strip().replace('$', '').replace(',', '')
                metrics['avg_loss'] = float(avg_loss_str)
            elif 'Profit Factor:' in line:
                pf_str = line.split(':')[1].strip()
                try:
                    # Handle infinity symbol or "no losses" text
                    if '∞' in pf_str or 'no losses' in pf_str.lower():
                        metrics['profit_factor'] = 999.99  # Very high value to indicate perfect strategy
                    else:
                        metrics['profit_factor'] = float(pf_str)
                except:
                    metrics['profit_factor'] = 0.0
            elif 'Expectancy:' in line:
                exp_str = line.split(':')[1].strip().replace('$', '').replace(',', '')
                try:
                    metrics['expectancy'] = float(exp_str)
                except:
                    metrics['expectancy'] = 0.0

        return metrics

    except subprocess.TimeoutExpired:
        print(f"  Timeout running backtest with params ({sma_period}, {std_multiplier})")
        return None
    except Exception as e:
        print(f"  Error running backtest: {e}")
        return None


def optimize_parameters(train_file: str) -> List[Dict]:
    """
    Test different parameter combinations on training data

    Args:
        train_file: Path to training data CSV

    Returns:
        List of results sorted by total P&L
    """
    print(f"\n{'='*60}")
    print("PARAMETER OPTIMIZATION (Training Set)")
    print(f"{'='*60}\n")

    # Parameter ranges to test - comprehensive search
    # SMA periods: from 1 to 100 with strategic spacing
    sma_periods = (
        list(range(1, 11)) +           # 1-10: every value (high frequency)
        list(range(12, 21, 2)) +       # 12-20: every 2 (short-term)
        list(range(25, 51, 5)) +       # 25-50: every 5 (medium-term)
        list(range(60, 101, 10))       # 60-100: every 10 (long-term)
    )

    # Std multipliers: from 0.1 to 4.0 with fine granularity
    std_multipliers = (
        [round(x * 0.1, 1) for x in range(1, 11)] +    # 0.1-1.0: every 0.1
        [round(x * 0.25, 2) for x in range(5, 17)]     # 1.25-4.0: every 0.25
    )

    total_combinations = len(sma_periods) * len(std_multipliers)
    print(f"Testing {total_combinations} parameter combinations...")
    print(f"SMA periods: {len(sma_periods)} values from 1 to 100")
    print(f"  Range: {min(sma_periods)} to {max(sma_periods)}")
    print(f"Std multipliers: {len(std_multipliers)} values from 0.1 to 4.0")
    print(f"  Range: {min(std_multipliers)} to {max(std_multipliers)}")
    print()

    results = []
    tested = 0
    start_time = datetime.now()

    # Test all combinations
    for sma_period in sma_periods:
        for std_mult in std_multipliers:
            tested += 1

            # Progress indicator with ETA
            if tested > 1:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time_per_test = elapsed / (tested - 1)
                remaining = total_combinations - tested
                eta_seconds = int(avg_time_per_test * remaining)
                eta_minutes = eta_seconds // 60
                eta_seconds = eta_seconds % 60
                eta_str = f"ETA: {eta_minutes}m {eta_seconds}s"
            else:
                eta_str = "ETA: calculating..."

            print(f"[{tested}/{total_combinations}] SMA={sma_period:3d}, Std={std_mult:.2f} | {eta_str}...", end=' ')

            metrics = run_backtest(train_file, sma_period, std_mult)

            if metrics and metrics['total_trades'] > 0:
                results.append(metrics)
                print(f"P&L=${metrics['total_pnl']:>10.2f}, Trades={metrics['total_trades']:4d}, WR={metrics['win_rate']:5.1f}%, Exp=${metrics['expectancy']:6.2f}")
            else:
                print("No trades")

    # Sort by total P&L (descending)
    results.sort(key=lambda x: x['total_pnl'], reverse=True)

    # Calculate total time
    total_time = (datetime.now() - start_time).total_seconds()
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"\nTested {total_combinations} combinations in {minutes}m {seconds}s")
    print(f"Valid results: {len(results)}")
    print(f"\nTop 20 parameter combinations (by P&L):\n")
    print(f"{'Rank':<6} {'SMA':<6} {'Std':<7} {'P&L':<14} {'Trades':<8} {'Win%':<8} {'PF':<8} {'Expectancy':<12}")
    print(f"{'-'*78}")

    for i, result in enumerate(results[:20], 1):
        print(f"{i:<6} {result['sma_period']:<6} {result['std_multiplier']:<7.2f} "
              f"${result['total_pnl']:<13,.2f} {result['total_trades']:<8} "
              f"{result['win_rate']:<7.1f}% {result['profit_factor']:<8.2f} ${result['expectancy']:<11.2f}")

    return results


def validate_parameters(params_list: List[Dict], validation_file: str, test_file: str, top_n: int = 10) -> pd.DataFrame:
    """
    Validate top N parameters on validation and test sets

    Args:
        params_list: List of parameter results from optimization
        validation_file: Path to validation data
        test_file: Path to test data
        top_n: Number of top parameters to validate

    Returns:
        DataFrame with validation results
    """
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION (Top {top_n} Parameters)")
    print(f"{'='*60}\n")

    validation_results = []

    for i, params in enumerate(params_list[:top_n], 1):
        sma_period = params['sma_period']
        std_mult = params['std_multiplier']

        print(f"\n[{i}/{top_n}] Testing SMA={sma_period}, Std={std_mult}")
        print(f"{'-'*60}")

        # Run on validation set
        print(f"  Validation set (Q3)...", end=' ')
        val_metrics = run_backtest(validation_file, sma_period, std_mult)
        if val_metrics:
            print(f"P&L=${val_metrics['total_pnl']:.2f}, Trades={val_metrics['total_trades']}, WR={val_metrics['win_rate']:.1f}%")
        else:
            print("Failed")
            continue

        # Run on test set
        print(f"  Test set (Q4)...", end=' ')
        test_metrics = run_backtest(test_file, sma_period, std_mult)
        if test_metrics:
            print(f"P&L=${test_metrics['total_pnl']:.2f}, Trades={test_metrics['total_trades']}, WR={test_metrics['win_rate']:.1f}%")
        else:
            print("Failed")
            continue

        # Store results
        validation_results.append({
            'rank': i,
            'sma_period': sma_period,
            'std_multiplier': std_mult,
            'train_pnl': params['total_pnl'],
            'train_trades': params['total_trades'],
            'train_win_rate': params['win_rate'],
            'val_pnl': val_metrics['total_pnl'],
            'val_trades': val_metrics['total_trades'],
            'val_win_rate': val_metrics['win_rate'],
            'test_pnl': test_metrics['total_pnl'],
            'test_trades': test_metrics['total_trades'],
            'test_win_rate': test_metrics['win_rate'],
            'avg_pnl': (val_metrics['total_pnl'] + test_metrics['total_pnl']) / 2,
            'consistency': 1.0 if (val_metrics['total_pnl'] > 0 and test_metrics['total_pnl'] > 0) else 0.0
        })

    return pd.DataFrame(validation_results)


def print_final_results(results_df: pd.DataFrame):
    """
    Print final validation results with analysis

    Args:
        results_df: DataFrame with validation results
    """
    print(f"\n{'='*60}")
    print("FINAL WALK-FORWARD VALIDATION RESULTS")
    print(f"{'='*60}\n")

    # Sort by average out-of-sample P&L
    results_df = results_df.sort_values('avg_pnl', ascending=False)

    print("Performance across all data splits:\n")
    print(f"{'Rank':<6} {'SMA':<6} {'Std':<6} {'Train P&L':<12} {'Val P&L':<12} {'Test P&L':<12} {'Avg OOS':<12} {'Consistent':<11}")
    print(f"{'-'*95}")

    for _, row in results_df.iterrows():
        consistent = "✓" if row['consistency'] == 1.0 else "✗"
        print(f"{int(row['rank']):<6} {int(row['sma_period']):<6} {row['std_multiplier']:<6.1f} "
              f"${row['train_pnl']:<11.2f} ${row['val_pnl']:<11.2f} ${row['test_pnl']:<11.2f} "
              f"${row['avg_pnl']:<11.2f} {consistent:<11}")

    # Find best parameter set
    best = results_df.iloc[0]

    print(f"\n{'='*60}")
    print("RECOMMENDED PARAMETERS")
    print(f"{'='*60}")
    print(f"\nBest parameters based on out-of-sample performance:")
    print(f"  SMA Period:        {int(best['sma_period'])}")
    print(f"  Std Multiplier:    {best['std_multiplier']:.1f}")
    print(f"\nPerformance:")
    print(f"  Training P&L:      ${best['train_pnl']:.2f} ({int(best['train_trades'])} trades)")
    print(f"  Validation P&L:    ${best['val_pnl']:.2f} ({int(best['val_trades'])} trades)")
    print(f"  Test P&L:          ${best['test_pnl']:.2f} ({int(best['test_trades'])} trades)")
    print(f"  Avg Out-of-Sample: ${best['avg_pnl']:.2f}")

    # Calculate consistency metrics
    consistent_params = results_df[results_df['consistency'] == 1.0]
    print(f"\nRobustness Analysis:")
    print(f"  Consistent performers: {len(consistent_params)}/{len(results_df)} ({len(consistent_params)/len(results_df)*100:.0f}%)")
    print(f"  (Positive P&L on both validation and test sets)")

    # Check for overfitting
    train_val_ratio = best['val_pnl'] / best['train_pnl'] if best['train_pnl'] != 0 else 0
    train_test_ratio = best['test_pnl'] / best['train_pnl'] if best['train_pnl'] != 0 else 0

    print(f"\nOverfitting Check:")
    print(f"  Validation/Training ratio: {train_val_ratio:.2f}")
    print(f"  Test/Training ratio:       {train_test_ratio:.2f}")

    if train_val_ratio < 0.3 or train_test_ratio < 0.3:
        print(f"  ⚠ WARNING: Possible overfitting detected!")
        print(f"  Out-of-sample performance is significantly worse than training.")
    elif train_val_ratio > 0.7 and train_test_ratio > 0.7:
        print(f"  ✓ Good generalization - parameters perform well on unseen data")
    else:
        print(f"  → Moderate generalization - acceptable but monitor carefully")

    print(f"\n{'='*60}")
    print("\nTo use the best parameters:")
    print(f"  ./backtest <data_file> {int(best['sma_period'])} {best['std_multiplier']:.1f}")
    print(f"{'='*60}\n")


def cleanup_temp_files(train_file: str, validation_file: str, test_file: str):
    """Remove temporary data split files"""
    for f in [train_file, validation_file, test_file]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed: {f}")


def find_best_main(dataset="/datasets/btc_data.csv"):

    csv_file = dataset

    # Verify file exists
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("MEAN REVERSION STRATEGY OPTIMIZER")
    print("Walk-Forward Optimization with Train/Validation/Test Splits")
    print(f"{'='*60}")
    print(f"Data file: {csv_file}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Step 1: Split data
        train_file, validation_file, test_file = split_data(csv_file)

        # Step 2: Optimize parameters on training set
        optimization_results = optimize_parameters(train_file)

        if not optimization_results:
            print("\nError: No valid results from optimization")
            cleanup_temp_files(train_file, validation_file, test_file)
            sys.exit(1)

        # Step 3: Validate top parameters on validation and test sets
        validation_df = validate_parameters(optimization_results, validation_file, test_file, top_n=10)

        # Step 4: Print final results and recommendations
        print_final_results(validation_df)

        # Save results to CSV
        # output_file = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # validation_df.to_csv(output_file, index=False)
        # print(f"Results saved to: {output_file}\n")

        # Cleanup
        print("\nCleaning up temporary files...")
        cleanup_temp_files(train_file, validation_file, test_file)

        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        print("\nCleaning up temporary files...")
        cleanup_temp_files(train_file, validation_file, test_file)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nCleaning up temporary files...")
        cleanup_temp_files(train_file, validation_file, test_file)
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python find_best.py <csv_file>")
        print("Example: python find_best.py btc_data.csv")
        sys.exit(1)

    find_best_main(sys.argv[1])
