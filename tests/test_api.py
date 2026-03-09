"""
tests/test_api.py
-----------------
Integration tests for the FastAPI prediction endpoints.

Uses FastAPI's TestClient (wraps httpx) so no real server is needed.
The model is mocked so tests don't depend on trained artefacts on disk.

MLOps role: Catches API contract breakage early.
            These tests are part of the CI pipeline and must pass
            before any Docker image is built or deployed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Sample customer payload – matches CustomerFeatures schema
# ---------------------------------------------------------------------------

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "seniorcitizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phoneservice": "Yes",
    "multiplelines": "No",
    "internetservice": "DSL",
    "onlinesecurity": "Yes",
    "onlinebackup": "No",
    "deviceprotection": "No",
    "techsupport": "No",
    "streamingtv": "No",
    "streamingmovies": "No",
    "contract": "Month-to-month",
    "paperlessbilling": "Yes",
    "paymentmethod": "Electronic check",
    "monthlycharges": 55.0,
    "totalcharges": 660.0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        # /health must always return 200, even when the model isn't loaded.
        # (Returning 503 would kill the pod before it has a chance to start.)
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_field_present(self):
        # Status is either "ok" (model loaded) or "degraded" (model missing).
        # Both are valid here because no real model file exists during tests.
        response = client.get("/health")
        assert response.json()["status"] in ("ok", "degraded")

    def test_health_has_readiness_flags(self):
        # The response must advertise whether the model and preprocessor
        # are loaded so K8s readiness probes can inspect the payload.
        data = client.get("/health").json()
        assert "model_loaded" in data
        assert "preprocessor_loaded" in data


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_model_name(self):
        response = client.get("/model-info")
        assert "model_name" in response.json()


class TestPredictEndpoint:
    def test_predict_with_mocked_model(self):
        """Mock predict_single so we don't need a real model file."""
        # PredictionResponse requires all four fields: churn, probability,
        # label, and threshold. Omitting threshold causes a 422 from Pydantic.
        mock_result = {"churn": 1, "probability": 0.82, "label": "Churn", "threshold": 0.5}
        with patch("api.main.predict_single", return_value=mock_result):
            response = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert response.status_code == 200
        data = response.json()
        assert data["churn"] == 1
        assert data["label"] == "Churn"
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_missing_field_returns_422(self):
        """Pydantic should reject payloads missing required fields."""
        incomplete = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "tenure"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_model_not_found_returns_503(self):
        """If model files are missing, API should return 503 (service unavailable)."""
        with patch("api.main.predict_single", side_effect=FileNotFoundError("no model")):
            response = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert response.status_code == 503


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_envelope(self):
        # /predict-batch returns {"scored": N, "predictions": [...]}
        # NOT a bare list — use the envelope to get the predictions array.
        mock_df_result = __import__("pandas").DataFrame([{
            **SAMPLE_CUSTOMER,
            "churn_probability": 0.75,
            "churn_prediction": 1,
            "churn_label": "Churn",
        }])
        with patch("api.main.predict_batch", return_value=mock_df_result):
            response = client.post(
                "/predict-batch",
                json={"customers": [SAMPLE_CUSTOMER]}
            )
        assert response.status_code == 200
        data = response.json()
        # Top-level envelope keys
        assert "scored" in data
        assert "predictions" in data
        # scored count matches input length
        assert data["scored"] == 1
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 1

    def test_batch_predict_missing_customers_returns_422(self):
        # Sending an empty body should fail Pydantic validation
        response = client.post("/predict-batch", json={})
        assert response.status_code == 422

    def test_batch_predict_empty_list_returns_422(self):
        # An empty customers list makes no sense to score
        response = client.post("/predict-batch", json={"customers": []})
        assert response.status_code == 422
