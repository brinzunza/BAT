"""Tiny FastAPI service: POST /predict {"bars":[{timestamp,Open,High,Low,Close,Volume},...]} -> score. Run: CHECKPOINT=checkpoints/model.pt uvicorn src.inference.api:app"""
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from .predict import predict_latest
app = FastAPI(title="BAT predictor")
@app.get("/health")
def health(): return {"ok": True, "checkpoint": os.environ.get("CHECKPOINT", "checkpoints/model.pt")}
@app.post("/predict")
def predict(payload: dict):
    try:
        d = pd.DataFrame(payload["bars"]); d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True); d = d.set_index("timestamp").astype(float)
        return {k: (str(v) if k == "as_of" else v) for k, v in predict_latest(d, os.environ.get("CHECKPOINT", "checkpoints/model.pt")).items()}
    except (KeyError, ValueError) as e: raise HTTPException(400, str(e))
