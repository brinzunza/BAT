import numpy as np, torch
from src.data.dataloader import build_loaders
from src.models.factory import build_model
from src.training.trainer import Trainer, predict_loader
from src.training.callbacks import EarlyStopping
from src.evaluation.metrics import sign_strategy_returns, sharpe
from src.evaluation.walk_forward import folds, walk_forward
from src.utils.checkpoint import save_checkpoint
from src.inference.predict import predict_latest
def test_trainer_runs_and_predicts(bars, cfgs):
    d, m, t = cfgs; D = build_loaders(d, bars); model = build_model(m, D["n_features"], d["seq_len"]); h = Trainer(model, t, torch.device("cpu")).fit(D["train"], D["val"])
    p, y = predict_loader(model, D["test"], torch.device("cpu")); assert len(h) >= 1 and len(p) == len(y) == len(D["test"].dataset) and np.isfinite(p).all()
def test_early_stopping_restores_best():
    m = torch.nn.Linear(1, 1); es = EarlyStopping(2); es.step(1.0, m); w = m.weight.item(); m.weight.data += 5; es.step(2.0, m); assert es.step(3.0, m); es.restore(m); assert m.weight.item() == w
def test_walk_forward_folds_never_overlap_train_and_test():
    for (a0, a1), (b0, b1) in folds(1000, 5, purge=4): assert a1 + 4 == b0 and b1 > b0
def test_sign_strategy_costs():
    r = sign_strategy_returns(np.array([1., 1., -1.]), np.array([.01, .01, .01]), cost=0.001); assert np.allclose(r, [.01 - .001, .01, -.01 - .002])
def test_walk_forward_smoke(bars, cfgs):
    d, m, t = cfgs; rows = walk_forward(bars, d, m, t, n_folds=2); assert len(rows) == 2 and all(np.isfinite(r["acc"]) for r in rows)
def test_checkpoint_roundtrip_and_predict(bars, cfgs, tmp_path):
    d, m, t = cfgs; D = build_loaders(d, bars); model = build_model(m, D["n_features"], d["seq_len"]); cfg = dict(data=d, model=m, train=t, n_features=D["n_features"])
    save_checkpoint(tmp_path / "m.pt", model, cfg, D["scaler"]); out = predict_latest(bars, str(tmp_path / "m.pt")); assert 0 <= out["prob_up"] <= 1 and out["as_of"] == bars.index[-1]
