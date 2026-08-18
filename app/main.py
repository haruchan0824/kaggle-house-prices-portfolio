from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.features import add_domain_features


MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
app = FastAPI(title="House Prices Prediction API", version="1.0.0")
_artifact: dict[str, Any] | None = None


def _load_artifact() -> dict[str, Any]:
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def _predict_log(row: pd.DataFrame, artifact: dict[str, Any]) -> float:
    name = artifact["recommended_model_name"]
    models = artifact["models"]
    if name in models:
        return float(models[name].predict(row)[0])
    weights = artifact["ensemble_weights"].get(name)
    if weights is None:
        raise ValueError(f"Unsupported recommended model: {name}")
    return float(sum(weight * models[model].predict(row)[0] for model, weight in weights.items()))


@app.get("/health")
def health() -> dict[str, str]:
    try:
        _load_artifact()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "model": "loaded"}


@app.post("/predict")
def predict(features: dict[str, Any]) -> dict[str, float | str]:
    if not features:
        raise HTTPException(status_code=422, detail="Request body must contain house features")
    try:
        artifact = _load_artifact()
        row = add_domain_features(pd.DataFrame([features]))
        row = row.reindex(columns=artifact["feature_columns"])
        prediction = float(np.expm1(_predict_log(row, artifact)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid feature payload: {exc}") from exc
    if not np.isfinite(prediction):
        raise HTTPException(status_code=500, detail="Model returned a non-finite prediction")
    return {"prediction": prediction, "currency": "USD"}
