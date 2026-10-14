import pytest, torch
from src.models.factory import build_model, MODELS
@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_forward_shape(name):
    m = build_model(dict(name=name, hidden=16, layers=1, dropout=0.1), n_features=7, seq_len=12); out = m(torch.randn(5, 12, 7)); assert out.shape == (5,) and torch.isfinite(out).all()
def test_unknown_model_rejected():
    with pytest.raises(ValueError): build_model(dict(name="nope"), 3, 4)
