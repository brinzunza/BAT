from .mlp import MLP
from .cnn import CNN
from .lstm import LSTM
from .transformer import Transformer
MODELS = dict(mlp=MLP, cnn=CNN, lstm=LSTM, transformer=Transformer)
def build_model(model_cfg, n_features, seq_len):
    cfg = dict(model_cfg); name = cfg.pop("name")
    if name not in MODELS: raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](n_features=n_features, seq_len=seq_len, **cfg)
