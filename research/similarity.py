#!/usr/bin/env python3
"""
Similarity Analysis Script
Analyzes and visualizes the similarity of price movements across different assets
with the same timeframe and date range.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse CSV filename to extract metadata.
    Expected format: X_SYMBOL_TIMEFRAME_STARTDATE_to_ENDDATE.csv

    Args:
        filename: Name of the CSV file

    Returns:
        Dictionary containing parsed metadata
    """
    pattern = r'X_(.+?)_(.+?)_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.csv'
    match = re.match(pattern, filename)

    if match:
        return {
            'symbol': match.group(1),
            'timeframe': match.group(2),
            'start_date': match.group(3),
            'end_date': match.group(4),
            'group_key': f"{match.group(2)}_{match.group(3)}_to_{match.group(4)}"
        }
    return None

def load_and_prepare_data(file_path: str, symbol: str) -> pd.DataFrame:
    """
    Load CSV file and prepare data for analysis.

    Args:
        file_path: Path to the CSV file
        symbol: Asset symbol name

    Returns:
        DataFrame with timestamp index and returns column
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Calculate returns (percentage change)
    df['returns'] = df['Close'].pct_change()

    # Add normalized close price (for visual comparison)
    df['normalized_close'] = (df['Close'] / df['Close'].iloc[0]) * 100

    return df[['Close', 'returns', 'normalized_close']]


def calculate_similarity_metrics(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate correlation matrix between different assets.

    Args:
        data_dict: Dictionary mapping symbols to their DataFrames

    Returns:
        Correlation matrix as DataFrame
    """
    # Create a combined DataFrame of returns
    returns_dict = {}
    for symbol, df in data_dict.items():
        returns_dict[symbol] = df['returns']

    returns_df = pd.DataFrame(returns_dict)

    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()

    return correlation_matrix


def plot_normalized_prices(data_dict: Dict[str, pd.DataFrame], group_key: str, output_dir: Path):
    """
    Plot normalized prices for visual comparison.

    Args:
        data_dict: Dictionary mapping symbols to their DataFrames
        group_key: String identifying the group (timeframe and date range)
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(14, 7))

    for symbol, df in data_dict.items():
        plt.plot(df.index, df['normalized_close'], label=symbol, linewidth=1.5, alpha=0.8)

    plt.title(f'Normalized Price Movement Comparison\n({group_key})', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Price (Starting at 100)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f'normalized_prices_{group_key}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, group_key: str, output_dir: Path):
    """
    Plot correlation heatmap.

    Args:
        correlation_matrix: Correlation matrix DataFrame
        group_key: String identifying the group (timeframe and date range)
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation Coefficient'}
    )

    plt.title(f'Price Returns Correlation Matrix\n({group_key})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'correlation_heatmap_{group_key}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def print_summary_stats(data_dict: Dict[str, pd.DataFrame], correlation_matrix: pd.DataFrame):
    """
    Print summary statistics for the assets.

    Args:
        data_dict: Dictionary mapping symbols to their DataFrames
        correlation_matrix: Correlation matrix DataFrame
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for symbol, df in data_dict.items():
        print(f"\n{symbol}:")
        print(f"  Start Price: ${df['Close'].iloc[0]:,.2f}")
        print(f"  End Price: ${df['Close'].iloc[-1]:,.2f}")
        print(f"  Total Return: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")
        print(f"  Volatility (std of returns): {df['returns'].std() * 100:.3f}%")
        print(f"  Mean Return: {df['returns'].mean() * 100:.4f}%")

    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    print("\nCorrelation Matrix:")
    print(correlation_matrix.to_string())

    # Find most and least correlated pairs
    correlations = []
    symbols = correlation_matrix.columns.tolist()
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            correlations.append({
                'pair': f"{symbols[i]} - {symbols[j]}",
                'correlation': correlation_matrix.iloc[i, j]
            })

    if correlations:
        correlations_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        print(f"\nMost correlated pair: {correlations_df.iloc[0]['pair']} "
              f"(r = {correlations_df.iloc[0]['correlation']:.3f})")
        print(f"Least correlated pair: {correlations_df.iloc[-1]['pair']} "
              f"(r = {correlations_df.iloc[-1]['correlation']:.3f})")

    print("\n" + "="*80)


def main():
    """Main execution function."""
    # Set up paths
    datasets_dir = Path(__file__).parent / 'datasets'
    output_dir = Path(__file__).parent / 'similarity_results'

    if not datasets_dir.exists():
        print(f"Error: Dataset directory not found at {datasets_dir}")
        return

    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Find all CSV files
    csv_files = list(datasets_dir.glob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in {datasets_dir}")
        return

    print(f"Found {len(csv_files)} CSV files\n")

    # Group files by timeframe and date range
    groups = {}
    for csv_file in csv_files:
        metadata = parse_filename(csv_file.name)
        if metadata:
            group_key = metadata['group_key']
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append((csv_file, metadata))

    # Process each group
    for group_key, files in groups.items():
        print(f"{'='*80}")
        print(f"Processing group: {group_key}")
        print(f"Number of assets: {len(files)}")
        print(f"{'='*80}\n")

        # Load data for all files in the group
        data_dict = {}
        for file_path, metadata in files:
            symbol = metadata['symbol']
            print(f"  Loading {symbol}...", end=' ')
            data_dict[symbol] = load_and_prepare_data(file_path, symbol)
            print("Done")

        # Calculate similarity metrics
        print("\n  Calculating correlation metrics...", end=' ')
        correlation_matrix = calculate_similarity_metrics(data_dict)
        print("Done")

        # Print summary statistics
        print_summary_stats(data_dict, correlation_matrix)

        # Create visualizations
        print("\nGenerating visualizations:")
        plot_normalized_prices(data_dict, group_key, output_dir)
        plot_correlation_heatmap(correlation_matrix, group_key, output_dir)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
