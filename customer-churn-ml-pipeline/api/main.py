"""
api/main.py
-----------
FastAPI application that exposes the trained churn model as a REST service.

Endpoints:
  GET  /health       → Liveness probe (used by Docker / K8s health checks).
  GET  /model-info   → Returns the model name that is currently loaded.
  POST /predict      → Predict churn for a single customer.
  POST /predict-batch → Score a list of customers (batch endpoint).

ML role       : Makes the model consumable by any application via HTTP.
Data Eng role : Validates request schema (Pydantic) before touching the model.
MLOps role    : Health check + structured JSON responses support monitoring,
                load balancers, and CI smoke-tests.

--- Run locally ---
  uvicorn api.main:app --reload --port 8000
"""

import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import the predict module (path may differ depending on how you run the app)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.predict import predict_single, predict_batch
from src.utils import setup_logging

import pandas as pd

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Pydantic schemas – define the exact shape of request/response bodies.
# Pydantic validates types automatically so bad input is rejected before
# it ever reaches the model.
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """
    Input schema for a single customer prediction.
    Fields mirror the Telco Customer Churn dataset columns.
    All values should be provided as they appear BEFORE preprocessing.
    """
    gender: str = Field(..., example="Male")
    seniorcitizen: int = Field(..., example=0, ge=0, le=1)
    partner: str = Field(..., example="Yes")
    dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=24, ge=0)
    phoneservice: str = Field(..., example="Yes")
    multiplelines: str = Field(..., example="No phone service")
    internetservice: str = Field(..., example="Fiber optic")
    onlinesecurity: str = Field(..., example="No")
    onlinebackup: str = Field(..., example="Yes")
    deviceprotection: str = Field(..., example="No")
    techsupport: str = Field(..., example="No")
    streamingtv: str = Field(..., example="No")
    streamingmovies: str = Field(..., example="No")
    contract: str = Field(..., example="Month-to-month")
    paperlessbilling: str = Field(..., example="Yes")
    paymentmethod: str = Field(..., example="Electronic check")
    monthlycharges: float = Field(..., example=70.35, ge=0)
    totalcharges: float = Field(..., example=1687.0, ge=0)

    class Config:
        # Allow extra fields (future-proofing for schema changes)
        extra = "ignore"


class PredictionResponse(BaseModel):
    churn: int
    probability: float
    label: str


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Ops"])
def health_check() -> Dict[str, str]:
    """
    Liveness probe.  Returns 200 OK if the service is running.
    """
    return {"status": "ok"}


@app.get("/model-info", tags=["Ops"])
def model_info() -> Dict[str, Any]:
    """
    Returns metadata about the currently loaded model.
    """
    model_name = os.getenv("MODEL_NAME", "best_model")
    return {"model_name": model_name, "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """
    Predict churn for a single customer.

    Accepts a JSON body matching CustomerFeatures schema.
    Returns churn label (0/1), probability, and human-readable label.
    """
    try:
        input_dict = customer.model_dump()
        result = predict_single(input_dict)
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch", tags=["Prediction"])
def predict_batch_endpoint(request: BatchRequest) -> List[PredictionResponse]:
    """
    Score a batch of customers in one API call.

    Accepts a JSON body with a 'customers' array.
    Returns a list of prediction objects in the same order.
    """
    try:
        records = [c.model_dump() for c in request.customers]
        df = pd.DataFrame(records)
        result_df = predict_batch(df)

        responses = []
        for _, row in result_df.iterrows():
            churn_val = int(row["churn_probability"] >= 0.5)
            responses.append(PredictionResponse(
                churn=churn_val,
                probability=row["churn_probability"],
                label=row["churn_label"],
            ))
        return responses
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
    except Exception as e:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(e))
