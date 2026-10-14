"""python evaluate.py --ckpt checkpoints/model.pt [--walk-forward] [--cost 0.0001]   (test split metrics, or expanding-window walk-forward)"""
import argparse, yaml, numpy as np, torch
from src.data.dataloader import build_loaders
from src.data.preprocess import load_ohlcv
from src.evaluation import metrics as M
from src.evaluation.walk_forward import walk_forward
from src.inference.predict import load_model
from src.training.trainer import predict_loader
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints/model.pt"); ap.add_argument("--walk-forward", action="store_true"); ap.add_argument("--cost", type=float, default=0.0); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--data", default="configs/data.yaml"); ap.add_argument("--model", default="configs/model.yaml"); ap.add_argument("--train", default="configs/training.yaml"); a = ap.parse_args(argv)
    if a.walk_forward:
        dc, mc, tc = (yaml.safe_load(open(p)) for p in (a.data, a.model, a.train)); rows = walk_forward(load_ohlcv(dc["path"], dc.get("resample")), dc, mc, tc, a.folds, a.cost)
        for r in rows: print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
        return rows
    m, ck = load_model(a.ckpt); cfg = ck["cfg"]; D = build_loaders(cfg["data"]); p, y = predict_loader(m, D["test"], torch.device("cpu"))
    out = dict(n=len(p), acc=M.directional_accuracy(p, y, cfg["data"]["target"] == "direction")); print(out); return out
if __name__ == "__main__": main()
