import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from features import Features

class FeatureClustering:
    def __init__(self, window_size=50, step_size=50):
        self.window_size = window_size
        self.step_size = step_size
        self.scaler = StandardScaler()
        self.kmeans = None
        self.df = None
        self.feature_matrix = None
        self.window_indices = []

    def extract_features(self, df):
        features_list = []
        window_indices = []
        skipped_count = 0

        for i in range(0, len(df) - self.window_size, self.step_size):
            end = i + self.window_size
            window_df = df.iloc[i:end].copy().reset_index(drop=True)
            featuresExtractor = Features(window_df, self.window_size)

            features = {
                'trend_rating': featuresExtractor.trend_rating(),
                'volatility': featuresExtractor.volatility(),
                'volume': featuresExtractor.volume(),
                'momentum': featuresExtractor.momentum(),
            }

            # Calculate volatility_std as std of rolling volatility values
            # Use a smaller rolling window to get multiple volatility values
            close_prices = window_df['Close']
            rolling_vol = close_prices.rolling(window=min(10, self.window_size)).std() / close_prices.mean()
            features['volatility_std'] = rolling_vol.std() if not rolling_vol.isna().all() else 0.0

            features['price_range'] = (window_df['High'].max() - window_df['Low'].min()) / window_df['Close'].mean()

            # Skip this window if any feature is NaN or infinite
            if any(pd.isna(v) or np.isinf(v) for v in features.values()):
                skipped_count += 1
                if skipped_count <= 3:  # Print first few skips for debugging
                    print(f"Skipping window {i}-{end}, features: {features}")
                continue

            features_list.append(features)
            window_indices.append((i, end))

        if skipped_count > 0:
            print(f"Skipped {skipped_count} windows due to NaN/inf values")

        self.window_indices = window_indices
        feature_df = pd.DataFrame(features_list)

        print(f"Created {len(feature_df)} feature vectors")

        # Final check: remove any rows with NaN
        feature_df = feature_df.dropna()

        return feature_df.values, feature_df.columns.tolist()

    def optimal_clusters(self, df, max_clusters=8):
        self.df = df
        feature_matrix, feature_names = self.extract_features(df)
        self.feature_matrix = feature_matrix

        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)

        results = {
            'n_clusters': [], 
            'intertia': [], 
            'silhouette': [], 
        }

        print(f"Finding optimal number of clusters...")
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=1, n_init=10)
            labels = kmeans.fit_predict(feature_matrix_scaled)
            inertia = kmeans.inertia_
            silhouette = silhouette_score(feature_matrix_scaled, labels)

            results['n_clusters'].append(k)
            results['intertia'].append(inertia)
            results['silhouette'].append(silhouette)

            if silhouette > 0.5:
                quality = "Excellent"
            elif silhouette > 0.3:
                quality = "Good"
            elif silhouette > 0.2:
                quality = "Fair"
            else:
                quality = "Poor"

            print(f"  k={k:2d}: Inertia={inertia:.2f}, Silhouette={silhouette:.3f} ({quality})")

        best_k = results['n_clusters'][np.argmax(results['silhouette'])]
        print(f"Optimal number of clusters: {best_k}")
        return pd.DataFrame(results)

    def fit(self, csv_file, n_clusters=3):
        df = pd.read_csv(csv_file)

        # Ensure columns are correctly named (CSV has: Volume,Open,Close,High,Low,timestamp)
        required_cols = ['Volume', 'Open', 'Close', 'High', 'Low']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}. CSV must have: Volume, Open, Close, High, Low, timestamp")

        self.df = df

        print(f"Extracting features from {len(df)} bars...")

        feature_matrix, feature_names = self.extract_features(df)
        self.feature_matrix = feature_matrix
        self.feature_names = feature_names

        print(f"Created {len(feature_matrix)} windows with {len(feature_names)} features each")

        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)

        print(f"Fitting K-Means with {n_clusters} clusters...")

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init=10)
        self.kmeans.fit(feature_matrix_scaled)

        self._analyze_clusters()

        return self

    def _analyze_clusters(self):
        """Analyze and print cluster characteristics"""
        labels = self.kmeans.labels_

        print(f"\nCluster Analysis:")
        print(f"{'Cluster':<10} {'Count':<10} {'Percentage':<12}")
        print("-" * 32)

        for cluster_id in range(self.kmeans.n_clusters):
            count = np.sum(labels == cluster_id)
            percentage = (count / len(labels)) * 100
            print(f"{cluster_id:<10} {count:<10} {percentage:>6.2f}%")

        # Print feature centroids for each cluster
        print(f"\nCluster Centroids (scaled):")
        feature_matrix_scaled = self.scaler.transform(self.feature_matrix)

        for cluster_id in range(self.kmeans.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_features = self.feature_matrix[cluster_mask]

            print(f"\nCluster {cluster_id}:")
            for i, feature_name in enumerate(self.feature_names):
                mean_val = cluster_features[:, i].mean()
                print(f"  {feature_name}: {mean_val:.4f}")