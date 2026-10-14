from pathlib import Path
import torch
def save_checkpoint(path, model, cfg, scaler=None, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), cfg=cfg, scaler=scaler, extra=extra or {}), path)
def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)
