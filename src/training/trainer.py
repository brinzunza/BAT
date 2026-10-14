import torch
from .losses import get_loss
from .optimizers import get_optimizer
from .schedulers import get_scheduler
from .callbacks import EarlyStopping
from ..utils.logging import get_logger
log = get_logger("trainer")
@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval(); P, Y = [], []
    for x, y in loader: P.append(model(x.to(device)).cpu()); Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()
class Trainer:
    def __init__(s, model, cfg, device, run_logger=None):
        s.m, s.cfg, s.dev, s.rl = model.to(device), cfg, device, run_logger
        s.loss = get_loss(cfg["loss"]); s.opt = get_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg.get("weight_decay", 0.0)); s.sched = get_scheduler(cfg.get("scheduler"), s.opt, cfg["epochs"])
    def _epoch(s, loader, train):
        s.m.train(train); tot = n = 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(s.dev), y.to(s.dev); l = s.loss(s.m(x), y)
                if train:
                    s.opt.zero_grad(); l.backward()
                    if s.cfg.get("grad_clip"): torch.nn.utils.clip_grad_norm_(s.m.parameters(), s.cfg["grad_clip"])
                    s.opt.step()
                tot += l.item() * len(x); n += len(x)
        return tot / max(n, 1)
    def fit(s, train_loader, val_loader):
        es = EarlyStopping(s.cfg.get("patience", 6)); hist = []
        for ep in range(s.cfg["epochs"]):
            tr, va = s._epoch(train_loader, True), s._epoch(val_loader, False); hist.append((tr, va))
            if s.sched is not None: s.sched.step(va) if s.cfg.get("scheduler") == "plateau" else s.sched.step()
            if s.rl: s.rl.log(epoch=ep, train_loss=tr, val_loss=va)
            log.info("epoch %d train %.4f val %.4f", ep, tr, va)
            if es.step(va, s.m): break
        es.restore(s.m); return hist
