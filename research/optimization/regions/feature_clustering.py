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
            features = 