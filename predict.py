"""python predict.py --ckpt checkpoints/model.pt --bars path/to/recent_bars.csv   -> score for the latest closed bar"""
import argparse
from src.data.preprocess import load_ohlcv
from src.inference.predict import predict_latest
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints/model.pt"); ap.add_argument("--bars", required=True); ap.add_argument("--resample", default=None); a = ap.parse_args()
    print(predict_latest(load_ohlcv(a.bars, a.resample), a.ckpt))
