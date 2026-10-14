import torch.nn as nn
class LSTM(nn.Module):
    def __init__(s, n_features, hidden=64, layers=2, dropout=0.2, **_):
        super().__init__(); s.rnn = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0); s.head = nn.Linear(hidden, 1)
    def forward(s, x): o, _h = s.rnn(x); return s.head(o[:, -1]).squeeze(-1)
