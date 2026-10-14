import torch.nn as nn
class CNN(nn.Module):
    def __init__(s, n_features, hidden=64, layers=2, dropout=0.2, kernel=5, **_):
        super().__init__(); mods = []; c = n_features
        for _i in range(layers): mods += [nn.Conv1d(c, hidden, kernel, padding=kernel // 2), nn.GELU(), nn.Dropout(dropout)]; c = hidden
        s.conv = nn.Sequential(*mods); s.head = nn.Linear(hidden, 1)
    def forward(s, x): return s.head(s.conv(x.transpose(1, 2)).mean(-1)).squeeze(-1)
