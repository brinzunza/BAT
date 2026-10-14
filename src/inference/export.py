import torch
from .predict import load_model
def export_torchscript(ckpt_path, out_path):
    m, ck = load_model(ckpt_path); L, F = ck["cfg"]["data"]["seq_len"], ck["cfg"]["n_features"]
    torch.jit.trace(m, torch.zeros(1, L, F)).save(out_path); return out_path
def export_onnx(ckpt_path, out_path):
    m, ck = load_model(ckpt_path); L, F = ck["cfg"]["data"]["seq_len"], ck["cfg"]["n_features"]
    torch.onnx.export(m, torch.zeros(1, L, F), out_path, input_names=["x"], output_names=["score"], dynamic_axes={"x": {0: "batch"}}); return out_path
