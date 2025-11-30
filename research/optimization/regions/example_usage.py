"""
Example usage of FeatureClustering and Features classes

CSV Format Expected:
Volume,Open,Close,High,Low,timestamp
"""
from feature_clustering import FeatureClustering
import pandas as pd

# Replace with your actual CSV file path
csv_file = "/Users/brunoinzunza/Documents/GitHub/BAT/research/datasets/X_BTCUSD_minute_2025-01-01_to_2025-09-01.csv"  # Must have columns: Volume, Open, Close, High, Low, timestamp

# Example 1: Find optimal number of clusters
print("=== Finding Optimal Number of Clusters ===")
clustering = FeatureClustering(window_size=50, step_size=50)

# Load data and find optimal clusters
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} rows from {csv_file}")
print(f"Columns: {df.columns.tolist()}")

results_df = clustering.optimal_clusters(df, max_clusters=8)
print("\nOptimal Clusters Results:")
print(results_df)

# Example 2: Fit with specific number of clusters
print("\n=== Fitting with Specific Number of Clusters ===")
clustering2 = FeatureClustering(window_size=50, step_size=50)
clustering2.fit(csv_file, n_clusters=3)

# Access cluster labels for each window
labels = clustering2.kmeans.labels_
print(f"\nCluster labels for each window: {labels}")

# Access window indices (start, end positions)
print(f"\nNumber of windows: {len(clustering2.window_indices)}")
print(f"First window covers rows: {clustering2.window_indices[0]}")

# Access feature matrix
print(f"\nFeature matrix shape: {clustering2.feature_matrix.shape}")
print(f"Features used: {clustering2.feature_names}")

# Example 3: Get cluster assignment for each timestamp
print("\n=== Mapping Clusters to Timestamps ===")
for i, (start, end) in enumerate(clustering2.window_indices[:5]):  # Show first 5
    cluster = labels[i]
    timestamp_start = df.iloc[start]['timestamp'] if 'timestamp' in df.columns else start
    timestamp_end = df.iloc[end-1]['timestamp'] if 'timestamp' in df.columns else end-1
    print(f"Window {i}: Rows {start}-{end}, Cluster {cluster}, Timestamps: {timestamp_start} to {timestamp_end}")
