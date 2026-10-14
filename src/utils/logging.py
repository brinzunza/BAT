import json, logging, time
from pathlib import Path
def get_logger(name="bat", level=logging.INFO):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")); lg.addHandler(h)
    lg.setLevel(level); return lg
class RunLogger:
    """Append-only JSONL metrics under runs/<name>/metrics.jsonl (one line per epoch/fold)."""
    def __init__(self, run_dir="runs", name=None):
        self.dir = Path(run_dir) / (name or time.strftime("%Y%m%d_%H%M%S")); self.dir.mkdir(parents=True, exist_ok=True)
    def log(self, **kw):
        with open(self.dir / "metrics.jsonl", "a") as f: f.write(json.dumps(kw, default=float) + "\n")
