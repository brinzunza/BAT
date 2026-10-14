import numpy as np, pandas as pd, pytest
@pytest.fixture
def bars():
    rng = np.random.RandomState(0); n = 3000; c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n))); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.0005, n)); l = np.minimum(o, c) * (1 - rng.uniform(0, 0.0005, n))
    return pd.DataFrame(dict(Open=o, High=h, Low=l, Close=c, Volume=rng.uniform(1, 10, n)), index=pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"))
@pytest.fixture
def cfgs():
    return (dict(path=None, resample=None, seq_len=16, horizon=4, target="direction", split=dict(train=0.7, val=0.15, test=0.15), embargo=4, batch_size=64),
            dict(name="lstm", hidden=16, layers=1, dropout=0.1), dict(epochs=2, lr=1e-3, weight_decay=0.0, loss="bce", optimizer="adamw", scheduler="cosine", patience=3, grad_clip=1.0, seed=1, device="cpu"))
