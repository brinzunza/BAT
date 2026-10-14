import torch
def get_optimizer(name, params, lr, weight_decay=0.0):
    return dict(adam=torch.optim.Adam, adamw=torch.optim.AdamW, sgd=lambda p, **k: torch.optim.SGD(p, momentum=0.9, **k))[name](params, lr=lr, weight_decay=weight_decay)
