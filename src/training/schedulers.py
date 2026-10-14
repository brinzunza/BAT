import torch
def get_scheduler(name, opt, epochs):
    if name in (None, "none"): return None
    if name == "cosine": return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if name == "step": return torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.3)
    if name == "plateau": return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.3)
    raise ValueError(name)
