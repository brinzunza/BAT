import torch.nn as nn
class MLP(nn.Module):
    def __init__(s, n_features, seq_len, hidden=64, layers=2, dropout=0.2, **_):
        super().__init__(); d = n_features * seq_len; mods = []
        for _i in range(layers): mods += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]; d = hidden
        s.net = nn.Sequential(nn.Flatten(), *mods, nn.Linear(d, 1))
    def forward(s, x): return s.net(x).squeeze(-1)
