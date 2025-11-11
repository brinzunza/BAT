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

        for i in range(0, len(df) - self.window_size, self.step_size):
            end = i + self.window_size
            window_df = df.iloc[i:end].copy()
            featuresExtractor = Features(window_df, self.window_size)
            
            features = {
                'trend_rating': featuresExtractor.trend_rating(),
                'volatility': featuresExtractor.volatility(),
                'volume': featuresExtractor.volume(),
                'momentum': featuresExtractor.momentum(),
            }

            features['volatility_std'] = featureExtractor.volatility.std()
            features['price_range'] = (window_df['High'].max() - window_df['Low'].min()) / window_df['Close'].mean()
            
            features_list.append(features)
            window_indices.append((i, end))

        self.window_indices = window_indices
        feature_df = pd.DataFrame(features_list)
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