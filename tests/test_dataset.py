import numpy as np
from src.data.features import build_features
from src.data.preprocess import make_xy, chrono_split
from src.data.dataset import WindowDataset
from src.data.dataloader import build_loaders
def test_features_are_causal(bars):
    full = build_features(bars); cut = build_features(bars.iloc[:2000])
    assert np.allclose(full.iloc[:2000].values, cut.values, equal_nan=True)       # truncating the future changes nothing
def test_labels_use_only_future_after_t(bars):
    X, y = make_xy(bars, 4, "return"); t = X.index[100]; i = bars.index.get_loc(t)
    assert np.isclose(y.loc[t], np.log(bars.Close.iloc[i + 4] / bars.Close.iloc[i]), atol=1e-6)
def test_split_has_embargo_and_no_overlap():
    (a0, a1), (b0, b1), (c0, c1) = chrono_split(1000, dict(train=.7, val=.15, test=.15), embargo=4)
    assert a1 + 4 == b0 and b1 + 4 == c0 and c1 == 1000
def test_window_dataset_shapes():
    ds = WindowDataset(np.random.randn(50, 3), np.arange(50), 10); x, y = ds[0]; assert len(ds) == 41 and x.shape == (10, 3) and y == 9
def test_loaders_scaler_fit_on_train_only(bars, cfgs):
    D = build_loaders(cfgs[0], bars); assert D["n_features"] > 5 and len(D["train"].dataset) > 0 and len(D["test"].dataset) > 0
    assert D["index"]["train"][-1] < D["index"]["val"][0] < D["index"]["test"][0]
