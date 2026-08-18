import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main


class ConstantModel:
    def predict(self, rows):
        return np.full(len(rows), np.log1p(200_000.0))


def test_health_and_prediction(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "_artifact",
        {
            "feature_columns": ["OverallQual", "GrLivArea", "OverallQual_x_GrLivArea"],
            "recommended_model_name": "catboost",
            "models": {"catboost": ConstantModel()},
            "ensemble_weights": {},
        },
    )
    client = TestClient(main.app)
    assert client.get("/health").json() == {"status": "ok", "model": "loaded"}
    response = client.post("/predict", json={"OverallQual": 7, "GrLivArea": 1500})
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(200_000.0)
    assert client.post("/predict", json={}).status_code == 422
