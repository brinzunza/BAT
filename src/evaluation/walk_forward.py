"""Expanding-window walk-forward over a bars frame: train on the past (minus a purge of `horizon` bars), test on the next block, retrain each fold."""
import numpy as np
import torch
from ..data.preprocess import make_xy, Scaler
from ..data.dataset import WindowDataset
from ..models.factory import build_model
from ..training.trainer import Trainer, predict_loader
from ..utils.device import get_device
from .metrics import directional_accuracy, information_coefficient, sign_strategy_returns, sharpe
from torch.utils.data import DataLoader
def folds(n, n_folds=5, min_train=0.4, purge=4):
    first = int(n * min_train); size = (n - first) // n_folds
    for k in range(n_folds):
        te0 = first + k * size; te1 = n if k == n_folds - 1 else te0 + size; yield (0, te0 - purge), (te0, te1)
def walk_forward(bars, data_cfg, model_cfg, train_cfg, n_folds=5, cost=0.0):
    X, y = make_xy(bars, data_cfg["horizon"], "return"); yd = (y > 0).astype("float32"); target = yd if data_cfg["target"] == "direction" else y
    L, dev, rows = data_cfg["seq_len"], get_device(train_cfg.get("device", "auto")), []
    for k, ((a0, a1), (b0, b1)) in enumerate(folds(len(X), n_folds, purge=data_cfg["horizon"])):
        cut = a0 + int((a1 - a0) * 0.85); sc = Scaler().fit(X.values[a0:cut]); Z = sc.transform(X.values)
        mk = lambda s, e, sh: DataLoader(WindowDataset(Z[s:e], target.values[s:e], L), batch_size=data_cfg["batch_size"], shuffle=sh)
        m = build_model(model_cfg, Z.shape[1], L); Trainer(m, train_cfg, dev).fit(mk(a0, cut, True), mk(cut + data_cfg["horizon"], a1, False))
        p, _ = predict_loader(m, mk(b0, b1, False), dev); fwd = y.values[b0:b1][L - 1:]
        r = sign_strategy_returns(p, fwd, cost); rows.append(dict(fold=k, n=len(p), acc=directional_accuracy(p, yd.values[b0:b1][L - 1:]), ic=information_coefficient(p, fwd), sharpe=sharpe(r), mean_ret=float(r.mean())))
    return rows
