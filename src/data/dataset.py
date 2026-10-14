import numpy as np, torch
from torch.utils.data import Dataset
class WindowDataset(Dataset):
    """Sample i = features[i : i+seq_len] (shape [seq_len, n_feat]) -> label of the LAST bar in the window."""
    def __init__(self, X, y, seq_len):
        self.X = torch.as_tensor(np.asarray(X, dtype="float32")); self.y = torch.as_tensor(np.asarray(y, dtype="float32")); self.L = seq_len
    def __len__(self): return max(len(self.X) - self.L + 1, 0)
    def __getitem__(self, i): return self.X[i:i + self.L], self.y[i + self.L - 1]
    @property
    def n_features(self): return self.X.shape[1]
