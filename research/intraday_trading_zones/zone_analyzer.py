"""
Intraday Trading Zone Analyzer

This module analyzes different time zones of the trading day to identify their
personality characteristics such as volatility, ranging behavior, trending, etc.
using machine learning and statistical methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import time, datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from scipy import stats


class TradingZoneAnalyzer:
    """
    Analyzes intraday trading zones to identify personality characteristics.
    """

    def __init__(self, zone_duration_minutes: int = 30):
        """
        Initialize the analyzer.

        Args:
            zone_duration_minutes: Duration of each time zone in minutes
        """
        self.zone_duration_minutes = zone_duration_minutes
        self.scaler = StandardScaler()
        self.zone_profiles = {}

    def create_time_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split trading day into time zones.

        Args:
            df: DataFrame with datetime index and OHLCV data

        Returns:
            DataFrame with added 'time_zone' column
        """
        df = df.copy()
        df['time'] = df.index.time
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute

        # Create zone labels based on time
        df['time_zone'] = (df['hour'] * 60 + df['minute']) // self.zone_duration_minutes

        return df

    def calculate_zone_features(self, zone_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistical features for a time zone.

        Args:
            zone_data: DataFrame containing data for a specific time zone

        Returns:
            Dictionary of calculated features
        """
        if len(zone_data) == 0:
            return {}

        features = {}

        # Volatility features
        if 'close' in zone_data.columns:
            returns = zone_data['close'].pct_change().dropna()
            features['volatility'] = returns.std()
            features['avg_return'] = returns.mean()
            features['return_skewness'] = stats.skew(returns) if len(returns) > 2 else 0
            features['return_kurtosis'] = stats.kurtosis(returns) if len(returns) > 3 else 0

        # Range features
        if 'high' in zone_data.columns and 'low' in zone_data.columns:
            features['avg_range'] = (zone_data['high'] - zone_data['low']).mean()
            features['avg_range_pct'] = ((zone_data['high'] - zone_data['low']) / zone_data['close']).mean()

        # Trending features
        if 'close' in zone_data.columns:
            # Linear regression slope
            x = np.arange(len(zone_data))
            if len(x) > 1:
                slope, _, r_value, _, _ = stats.linregress(x, zone_data['close'].values)
                features['trend_strength'] = abs(slope)
                features['trend_direction'] = np.sign(slope)
                features['trend_r_squared'] = r_value ** 2
            else:
                features['trend_strength'] = 0
                features['trend_direction'] = 0
                features['trend_r_squared'] = 0

        # Volume features (if available)
        if 'volume' in zone_data.columns:
            features['avg_volume'] = zone_data['volume'].mean()
            features['volume_volatility'] = zone_data['volume'].std()

        # Price action features
        if 'open' in zone_data.columns and 'close' in zone_data.columns:
            body_size = abs(zone_data['close'] - zone_data['open'])
            features['avg_body_size'] = body_size.mean()
            features['bullish_ratio'] = (zone_data['close'] > zone_data['open']).sum() / len(zone_data)

        return features

    def analyze_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze all time zones in the dataset.

        Args:
            df: DataFrame with datetime index and OHLCV data

        Returns:
            DataFrame with zone analysis results
        """
        df_zones = self.create_time_zones(df)

        zone_analysis = []

        for zone_id in df_zones['time_zone'].unique():
            zone_data = df_zones[df_zones['time_zone'] == zone_id]

            # Get time range for this zone
            start_time = zone_data.index.min().time()
            end_time = zone_data.index.max().time()

            # Calculate features
            features = self.calculate_zone_features(zone_data)

            if features:
                features['zone_id'] = zone_id
                features['start_time'] = start_time
                features['end_time'] = end_time
                features['num_samples'] = len(zone_data)

                zone_analysis.append(features)

        return pd.DataFrame(zone_analysis)

    def classify_zone_personality(self, zone_features: pd.DataFrame,
                                   n_clusters: int = 5) -> pd.DataFrame:
        """
        Classify zones into personality types using clustering.

        Args:
            zone_features: DataFrame with calculated zone features
            n_clusters: Number of personality clusters

        Returns:
            DataFrame with personality classifications
        """
        # Select features for clustering
        feature_cols = [col for col in zone_features.columns
                       if col not in ['zone_id', 'start_time', 'end_time', 'num_samples']]

        X = zone_features[feature_cols].fillna(0)

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Cluster zones
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        zone_features['personality_cluster'] = kmeans.fit_predict(X_scaled)

        # Interpret clusters based on feature values
        zone_features['personality_label'] = zone_features['personality_cluster'].apply(
            lambda x: self._interpret_cluster(x, zone_features, feature_cols)
        )

        return zone_features

    def _interpret_cluster(self, cluster_id: int, df: pd.DataFrame,
                          feature_cols: List[str]) -> str:
        """
        Interpret cluster characteristics to assign personality label.

        Args:
            cluster_id: The cluster ID
            df: DataFrame with all zone features
            feature_cols: List of feature column names

        Returns:
            Personality label string
        """
        cluster_data = df[df['personality_cluster'] == cluster_id][feature_cols]

        avg_features = cluster_data.mean()

        # Determine personality based on feature averages
        if avg_features.get('volatility', 0) > df['volatility'].median():
            if avg_features.get('trend_r_squared', 0) > 0.5:
                return f"Volatile Trending (Cluster {cluster_id})"
            else:
                return f"Volatile Ranging (Cluster {cluster_id})"
        else:
            if avg_features.get('trend_r_squared', 0) > 0.5:
                return f"Calm Trending (Cluster {cluster_id})"
            elif avg_features.get('avg_range_pct', 0) < df['avg_range_pct'].median():
                return f"Consolidation (Cluster {cluster_id})"
            else:
                return f"Moderate Ranging (Cluster {cluster_id})"

    def get_zone_summary(self, zone_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for each personality type.

        Args:
            zone_features: DataFrame with classified zones

        Returns:
            Summary DataFrame
        """
        summary = zone_features.groupby('personality_label').agg({
            'volatility': ['mean', 'std'],
            'avg_return': 'mean',
            'trend_strength': 'mean',
            'avg_range_pct': 'mean',
            'zone_id': 'count'
        }).round(4)

        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.rename(columns={'zone_id_count': 'num_zones'})

        return summary

    def analyze_full_dataset(self, df: pd.DataFrame,
                           n_clusters: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete analysis pipeline.

        Args:
            df: DataFrame with datetime index and OHLCV data
            n_clusters: Number of personality clusters

        Returns:
            Tuple of (zone_features, summary)
        """
        # Analyze zones
        zone_features = self.analyze_zones(df)

        # Classify personalities
        zone_features = self.classify_zone_personality(zone_features, n_clusters)

        # Generate summary
        summary = self.get_zone_summary(zone_features)

        return zone_features, summary
