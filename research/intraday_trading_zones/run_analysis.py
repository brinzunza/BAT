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
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
    elif file_path.suffix in ['.pkl', '.pickle']:
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    return df


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

    # Display results
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
    DATA_PATH = "path/to/your/data.csv"  # Update with actual data path
    ZONE_DURATION = 30  # minutes
    NUM_CLUSTERS = 5
    DATE_COLUMN = "datetime"  # or "date", "timestamp", etc.
    OUTPUT_DIR = "results"  # Where to save results
    SYMBOL = "SPY"  # Trading symbol

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
