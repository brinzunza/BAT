"""
Runner script for Intraday Trading Zone Analysis

This script provides a simple interface to run the trading zone analysis
with specific inputs and parameters.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from zone_analyzer import TradingZoneAnalyzer


def load_data(file_path: str, date_column: str = None) -> pd.DataFrame:
    """
    Load trading data from file.

    Args:
        file_path: Path to the data file (CSV, pickle, etc.)
        date_column: Name of the date column (if CSV)

    Returns:
        DataFrame with datetime index
    """
    file_path = Path(file_path)

    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)

        # Normalize column names to lowercase for consistency
        df.columns = df.columns.str.lower()

        # Convert date_column to lowercase if provided
        if date_column:
            date_column = date_column.lower()

        if date_column:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
    elif file_path.suffix in ['.pkl', '.pickle']:
        df = pd.read_pickle(file_path)
        # Normalize column names to lowercase for consistency
        df.columns = df.columns.str.lower()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    return df


def format_time_ranges(zone_features: pd.DataFrame) -> str:
    """
    Format zone features into readable time ranges grouped by strategy.

    Args:
        zone_features: DataFrame with zone analysis results

    Returns:
        Formatted string with strategies and their time ranges
    """
    output = []

    # Group zones by personality/strategy
    grouped = zone_features.groupby('personality_label')

    for strategy, group in grouped:
        # Sort by start time
        group_sorted = group.sort_values('zone_id')

        # Get time ranges
        time_ranges = []
        for _, row in group_sorted.iterrows():
            start = row['start_time'].strftime('%H:%M') if hasattr(row['start_time'], 'strftime') else str(row['start_time'])
            end = row['end_time'].strftime('%H:%M') if hasattr(row['end_time'], 'strftime') else str(row['end_time'])
            time_ranges.append(f"{start}-{end}")

        # Get average metrics for this strategy
        avg_vol = group['volatility'].mean()
        avg_range = group['avg_range_pct'].mean() * 100  # Convert to percentage
        avg_trend = group['trend_strength'].mean()

        output.append(f"\n  Strategy: {strategy}")
        output.append(f"  Time Ranges: {', '.join(time_ranges)}")
        output.append(f"  Characteristics:")
        output.append(f"    - Average Volatility: {avg_vol:.4f}")
        output.append(f"    - Average Range: {avg_range:.3f}%")
        output.append(f"    - Trend Strength: {avg_trend:.2f}")
        output.append("")

    return '\n'.join(output)


def run_zone_analysis(
    data_path: str,
    zone_duration_minutes: int = 30,
    n_clusters: int = 5,
    date_column: str = None,
    output_dir: str = None,
    symbol: str = "UNKNOWN"
):
    """
    Run the complete trading zone analysis.

    Args:
        data_path: Path to the trading data file
        zone_duration_minutes: Duration of each time zone in minutes
        n_clusters: Number of personality clusters to identify
        date_column: Name of the date column (for CSV files)
        output_dir: Directory to save results (optional)
        symbol: Trading symbol/ticker name for labeling

    Returns:
        Tuple of (zone_features, summary)
    """
    print(f"{'='*60}")
    print(f"Intraday Trading Zone Analysis")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Data file: {data_path}")
    print(f"Zone duration: {zone_duration_minutes} minutes")
    print(f"Number of personality clusters: {n_clusters}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    df = load_data(data_path, date_column)
    print(f"Loaded {len(df)} rows of data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}\n")

    # Initialize analyzer
    print("Initializing analyzer...")
    analyzer = TradingZoneAnalyzer(zone_duration_minutes=zone_duration_minutes)

    # Run analysis
    print("Analyzing trading zones...")
    zone_features, summary = analyzer.analyze_full_dataset(df, n_clusters=n_clusters)

    print(f"\nAnalysis complete!")
    print(f"Total zones analyzed: {len(zone_features)}")
    print(f"Unique personalities identified: {zone_features['personality_label'].nunique()}\n")

    # Display strategy recommendations
    print(f"{'='*60}")
    print(f"TRADING STRATEGIES BY TIME OF DAY - {symbol}")
    print(f"{'='*60}")
    print(format_time_ranges(zone_features))

    # Display detailed results
    print(f"{'='*60}")
    print("PERSONALITY SUMMARY")
    print(f"{'='*60}")
    print(summary)
    print(f"\n{'='*60}")
    print("ZONE DETAILS")
    print(f"{'='*60}")
    print(zone_features[['zone_id', 'start_time', 'end_time', 'personality_label',
                         'volatility', 'trend_strength', 'avg_range_pct']].to_string())

    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        zone_file = output_path / f"{symbol}_zone_analysis_{timestamp}.csv"
        summary_file = output_path / f"{symbol}_summary_{timestamp}.csv"

        zone_features.to_csv(zone_file, index=False)
        summary.to_csv(summary_file)

        print(f"\n{'='*60}")
        print(f"Results saved to:")
        print(f"  Zone details: {zone_file}")
        print(f"  Summary: {summary_file}")
        print(f"{'='*60}")

    return zone_features, summary


def main():
    """
    Main entry point for command-line execution.
    """
    # Example usage with default parameters
    # Modify these parameters as needed

    # CONFIGURATION
    DATA_PATH = "/Users/brunoinzunza/Documents/GitHub/BAT/research/datasets/X_BTCUSD_minute_2025-01-01_to_2025-09-01_test.csv"  # Relative path to datasets folder
    ZONE_DURATION = 30  # minutes
    NUM_CLUSTERS = 5
    DATE_COLUMN = "timestamp"  # Dataset uses 'timestamp' column
    OUTPUT_DIR = "results"  # Where to save results
    SYMBOL = "BTCUSD"  # Trading symbol

    try:
        zone_features, summary = run_zone_analysis(
            data_path=DATA_PATH,
            zone_duration_minutes=ZONE_DURATION,
            n_clusters=NUM_CLUSTERS,
            date_column=DATE_COLUMN,
            output_dir=OUTPUT_DIR,
            symbol=SYMBOL
        )

        print("\nAnalysis completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: Data file not found - {e}")
        print("\nPlease update the DATA_PATH variable in the script with your data file path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # You can also import this module and call run_zone_analysis() directly
    # with custom parameters instead of using main()

    # Example of programmatic usage:
    # from run_analysis import run_zone_analysis
    # zones, summary = run_zone_analysis(
    #     data_path="my_data.csv",
    #     zone_duration_minutes=15,
    #     n_clusters=4,
    #     symbol="AAPL"
    # )

    main()
