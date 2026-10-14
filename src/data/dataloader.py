from torch.utils.data import DataLoader
from .preprocess import load_ohlcv, make_xy, chrono_split, Scaler
from .dataset import WindowDataset
def build_loaders(cfg, bars=None):
    """cfg = configs/data.yaml dict. Returns dict(train/val/test loaders, scaler, n_features, index) -- index maps each split to its label timestamps."""
    bars = load_ohlcv(cfg["path"], cfg.get("resample")) if bars is None else bars
    X, y = make_xy(bars, cfg["horizon"], cfg["target"]); L = cfg["seq_len"]
    (a0, a1), (b0, b1), (c0, c1) = chrono_split(len(X), cfg["split"], cfg.get("embargo", 0), L)
    sc = Scaler().fit(X.values[a0:a1]); Xs = sc.transform(X.values); out = dict(scaler=sc, n_features=Xs.shape[1], index={})
    for name, (s, e), shuffle in (("train", (a0, a1), True), ("val", (b0, b1), False), ("test", (c0, c1), False)):
        ds = WindowDataset(Xs[s:e], y.values[s:e], L); out[name] = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle); out["index"][name] = X.index[s:e][L - 1:]
    return out
