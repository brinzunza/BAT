"""Raw OHLCV CSV -> clean bars -> causal features + forward-return labels -> chronological split + train-fit scaler."""
import numpy as np, pandas as pd
from .features import build_features
def load_ohlcv(path, resample=None):
    d = pd.read_csv(path); d.columns = [c.strip().capitalize() if c.lower() != "timestamp" else "timestamp" for c in d.columns]
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True); d = d.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    d = d[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    if resample: d = d.resample(resample, label="left", closed="left").agg(dict(Open="first", High="max", Low="min", Close="last", Volume="sum")).dropna(subset=["Open"])
    return d[(d.Close > 0) & (d.High >= d.Low)]
def make_xy(bars, horizon=4, target="direction"):
    """label[t] = log(close[t+h]/close[t]) (known only at t+h): rows whose label is unavailable are dropped. Features at t use bars <= t."""
    F = build_features(bars); fwd = np.log(bars.Close).shift(-horizon) - np.log(bars.Close)
    y = (fwd > 0).astype("float32") if target == "direction" else fwd.astype("float32")
    ok = F.notna().all(axis=1) & fwd.notna(); return F[ok], y[ok]
def chrono_split(n, split, embargo=0, seq_len=1):
    """Index ranges (train, val, test) with an embargo gap; windows are built inside each range so none straddles a boundary."""
    a = int(n * split["train"]); b = a + int(n * split["val"])
    return (0, a), (a + embargo, b), (b + embargo, n)
class Scaler:
    """Robust-ish standardiser fit on TRAIN rows only."""
    def fit(self, X): self.mu = np.nanmean(X, 0); self.sd = np.nanstd(X, 0) + 1e-8; return self
    def transform(self, X): return np.clip((X - self.mu) / self.sd, -6, 6).astype("float32")
