import numpy as np, torch
from ..data.preprocess import load_ohlcv
from ..data.features import build_features
from ..models.factory import build_model
from ..utils.checkpoint import load_checkpoint
def load_model(path, device="cpu"):
    ck = load_checkpoint(path, device); cfg = ck["cfg"]; m = build_model(cfg["model"], cfg["n_features"], cfg["data"]["seq_len"]); m.load_state_dict(ck["state"]); return m.eval().to(device), ck
@torch.no_grad()
def predict_latest(bars, ckpt_path, device="cpu"):
    """Score the most recent CLOSED bars. Returns dict(prob_up | pred_return, as_of)."""
    m, ck = load_model(ckpt_path, device); L = ck["cfg"]["data"]["seq_len"]; F = build_features(bars).dropna()
    if len(F) < L: raise ValueError(f"need >= {L} usable bars, have {len(F)}")
    x = torch.as_tensor(ck["scaler"].transform(F.values[-L:])).unsqueeze(0).to(device); out = float(m(x))
    return dict(as_of=F.index[-1], score=out, prob_up=float(1 / (1 + np.exp(-out))) if ck["cfg"]["data"]["target"] == "direction" else None)
