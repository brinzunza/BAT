import math, torch, torch.nn as nn
class Transformer(nn.Module):
    def __init__(s, n_features, seq_len, hidden=64, layers=2, dropout=0.2, heads=4, **_):
        super().__init__(); s.inp = nn.Linear(n_features, hidden); pe = torch.zeros(seq_len, hidden); pos = torch.arange(seq_len).unsqueeze(1); div = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000.0) / hidden))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)[:, : hidden // 2]; s.register_buffer("pe", pe)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout, batch_first=True, norm_first=True), layers); s.head = nn.Linear(hidden, 1)
    def forward(s, x): h = s.enc(s.inp(x) + s.pe[: x.shape[1]]); return s.head(h[:, -1]).squeeze(-1)
