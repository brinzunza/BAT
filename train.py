"""python train.py [--data configs/data.yaml --model configs/model.yaml --train configs/training.yaml --name run1 --override model.name=transformer]"""
import argparse, yaml
from src.data.dataloader import build_loaders
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import RunLogger
def load_cfgs(a):
    cfg = dict(data=yaml.safe_load(open(a.data)), model=yaml.safe_load(open(a.model)), train=yaml.safe_load(open(a.train)))
    for o in a.override:
        k, v = o.split("=", 1); sec, key = k.split("."); cfg[sec][key] = yaml.safe_load(v)
    return cfg
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="configs/data.yaml"); ap.add_argument("--model", default="configs/model.yaml"); ap.add_argument("--train", default="configs/training.yaml")
    ap.add_argument("--name", default=None); ap.add_argument("--override", nargs="*", default=[]); a = ap.parse_args(argv); cfg = load_cfgs(a)
    set_seed(cfg["train"]["seed"]); dev = get_device(cfg["train"]["device"]); D = build_loaders(cfg["data"]); cfg["n_features"] = D["n_features"]
    m = build_model(cfg["model"], D["n_features"], cfg["data"]["seq_len"]); rl = RunLogger(cfg["train"]["run_dir"], a.name)
    Trainer(m, cfg["train"], dev, rl).fit(D["train"], D["val"]); path = f"{cfg['train']['checkpoint_dir']}/{a.name or 'model'}.pt"
    save_checkpoint(path, m.cpu(), cfg, D["scaler"]); print("saved", path, "| metrics:", rl.dir / "metrics.jsonl")
if __name__ == "__main__": main()
