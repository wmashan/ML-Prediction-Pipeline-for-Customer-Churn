# Customer Churn ML Prediction Pipeline

> **End-to-end MLOps project** — from raw CSV to a containerised, tested, and monitored REST API.  
> Covers Data Engineering · ML Engineering · Model Serving · CI/CD · Experiment Tracking.

[![CI Pipeline](https://github.com/YOUR_USERNAME/ML-Prediction-Pipeline-for-Customer-Churn/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ML-Prediction-Pipeline-for-Customer-Churn/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.13-orange?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)

---

## Table of Contents

- [Business Problem](#business-problem)
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Running the API](#running-the-api)
- [Docker](#docker)
- [Running Tests](#running-tests)
- [Sample API Request / Response](#sample-api-request--response)
- [Key Highlights](#key-highlights)

---

## Business Problem

**Acquiring a new customer costs 5–7× more than retaining an existing one.**

Telecom companies lose thousands of customers every month to competitors. This project builds a machine learning system that:

1. Identifies customers who are **likely to cancel** their subscription in the next billing cycle.
2. Returns a **churn probability** (0–100 %) so retention teams can prioritise outreach.
3. Allows the **decision threshold to be adjusted** per campaign (e.g., cast a wider net at 35 % vs the default 50 %).

The dataset is the **IBM Telco Customer Churn** dataset (7,043 customers, 20 features). A positive prediction means the customer is at risk of churning.

---

## Project Overview

This is a **production-style MLOps pipeline** built step-by-step, covering every layer of a real ML system:

| Layer | What was built |
|---|---|
| **Data** | CSV ingestion, schema validation, cleaning, type casting |
| **Feature Engineering** | Tenure grouping, service counts, contract risk score, spend ratios |
| **Training** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost with hyperparameter search |
| **Experiment Tracking** | MLflow — parent/child runs, metric logging, model registry |
| **Serving** | FastAPI — single prediction, batch prediction, health check, configurable threshold |
| **Containerisation** | Multi-stage Docker build with production-only dependencies |
| **Testing** | pytest — unit tests for every module, mocked API integration tests |
| **CI/CD** | GitHub Actions — lint → test → Docker build on every push |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                      │
│                                                             │
│  Raw CSV  →  Ingest  →  Clean  →  Feature Eng  →  Train    │
│                                         │            │      │
│                                    Preprocessor   Model     │
│                                    (.joblib)    (.joblib)   │
│                                         │            │      │
│                                    MLflow Experiment        │
│                                    Tracking + Registry      │
└─────────────────────────────────────────────────────────────┘
                               │
                    models/preprocessor.joblib
                    models/best_model.joblib
                               │
┌─────────────────────────────────────────────────────────────┐
│                       SERVING PIPELINE                      │
│                                                             │
│  HTTP Request  →  FastAPI  →  Validate  →  Clean           │
│                                                 │           │
│                                          Feature Eng        │
│                                                 │           │
│                                        Preprocessor         │
│                                        .transform()         │
│                                                 │           │
│                                          Model              │
│                                        .predict_proba()     │
│                                                 │           │
│  HTTP Response ←  JSON  ←  Threshold  ←  Probability       │
└─────────────────────────────────────────────────────────────┘
```

**Key design principle:** The preprocessing pipeline (`preprocessor.joblib`) is fitted **once** during training and reused during inference. This guarantees that feature scaling and encoding are identical at train time and serve time — eliminating [training-serving skew](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew).

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| ML / Data | scikit-learn 1.4, XGBoost, pandas 2.2, NumPy |
| API Framework | FastAPI 0.111, Pydantic 2.7, uvicorn |
| Experiment Tracking | MLflow 2.13 |
| Model Persistence | joblib |
| Containerisation | Docker (multi-stage build) |
| Testing | pytest, pytest-cov, httpx |
| CI/CD | GitHub Actions |
| Code Quality | flake8 |

---

## Folder Structure

```
ML-Prediction-Pipeline-for-Customer-Churn/
│
├── src/                          # Core ML pipeline
│   ├── data_ingestion.py         # Load CSV → validated DataFrame
│   ├── preprocessing.py          # Clean, cast types, fit sklearn pipeline
│   ├── feature_engineering.py    # Derive tenure groups, service counts, etc.
│   ├── train.py                  # Train all models, MLflow tracking, save best
│   ├── evaluate.py               # Metrics, confusion matrix, ROC curve
│   ├── predict.py                # Inference: raw input → churn probability
│   └── utils.py                  # Logging, timer, config helpers
│
├── api/
│   └── main.py                   # FastAPI app (4 endpoints)
│
├── tests/
│   ├── test_data_ingestion.py    # CSV loading and schema validation
│   ├── test_preprocessing.py     # Cleaning and sklearn pipeline
│   ├── test_feature_engineering.py # All 5 feature functions
│   ├── test_predict.py           # Inference module (mocked artefacts)
│   └── test_api.py               # API endpoints (mocked model)
│
├── models/
│   ├── best_model.joblib         # Trained model (git-ignored, re-created by train.py)
│   └── preprocessor.joblib       # Fitted preprocessor (must match model)
│
├── data/
│   ├── raw/                      # Drop churn.csv here
│   └── processed/                # Auto-generated clean data
│
├── notebooks/
│   └── 01_eda.ipynb              # Exploratory Data Analysis
│
├── samples/
│   ├── predict_single.json       # Sample single-prediction payload
│   ├── predict_batch.json        # Sample batch payload (4 customers)
│   └── postman_collection.json   # Ready-to-import Postman collection
│
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions: lint → test → docker build
│
├── Dockerfile                    # Multi-stage production build
├── .dockerignore                 # Excludes dev files from build context
├── requirements.txt              # All deps including dev/test tools
├── requirements-prod.txt         # Runtime-only deps (used by Docker)
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (optional, for containerised runs)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ML-Prediction-Pipeline-for-Customer-Churn.git
cd ML-Prediction-Pipeline-for-Customer-Churn
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the **Telco Customer Churn** dataset from Kaggle:  
<https://www.kaggle.com/datasets/blastchar/telco-customer-churn>

Place the file at:

```
data/raw/churn.csv
```

---

## Training the Model

Run the full training pipeline from the project root:

```bash
python -m src.train
```

This will:
1. Load and validate `data/raw/churn.csv`
2. Clean the data and engineer features
3. Fit and compare four models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
4. Log all runs, metrics, and parameters to MLflow
5. Save the best model to `models/best_model.joblib`
6. Save the fitted preprocessor to `models/preprocessor.joblib`
7. Register the best model in the MLflow Model Registry as `churn_best_model`

Expected output (abridged):

```
INFO | Training: LogisticRegression
INFO | Training: RandomForest
INFO | Training: GradientBoosting
INFO | Training: XGBoost
INFO | Best model: XGBoost (ROC-AUC: 0.847)
INFO | Model saved → models/best_model.joblib
INFO | Preprocessor saved → models/preprocessor.joblib
```

---

## MLflow Experiment Tracking

Start the MLflow UI to browse all training runs:

```bash
mlflow ui --port 5000
```

Then open: <http://localhost:5000>

**What you'll find:**

| View | Description |
|---|---|
| Experiments → `churn_prediction` | All training sessions |
| Each run | Parameters, metrics (AUC, F1, precision, recall), model artefacts |
| Models → `churn_best_model` | Model Registry with versioned, tagged releases |

Every training run is saved under a **parent run** (`training_session`) with one **child run** per model. This makes it easy to compare all four models side-by-side using MLflow's built-in comparison charts.

---

## Running the API

Make sure `models/best_model.joblib` and `models/preprocessor.joblib` exist (run training first).

```bash
uvicorn api.main:app --reload --port 8000
```

Then open: <http://localhost:8000/docs> — the interactive Swagger UI is auto-generated by FastAPI.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — reports model/preprocessor load status |
| `GET` | `/model-info` | Readiness probe — returns model name, class, feature count |
| `POST` | `/predict` | Score a single customer |
| `POST` | `/predict-batch` | Score a list of customers in one request |

**Configurable threshold** — append `?threshold=0.35` to `/predict` to lower the decision boundary for high-recall campaigns:

```
POST /predict?threshold=0.35
```

---

## Docker

### Build the image

```bash
docker build -t churn-api .
```

The multi-stage Dockerfile installs only `requirements-prod.txt` in the final image (no pytest, flake8, or httpx), resulting in a smaller and more secure container.

### Run the container

```bash
docker run -p 8000:8000 churn-api
```

Pass runtime overrides with `-e`:

```bash
docker run -p 8000:8000 \
  -e MODEL_NAME=best_model \
  -e LOG_LEVEL=DEBUG \
  -v "$(pwd)/models:/app/models" \
  churn-api
```

> **Note:** If `models/` is not baked into the image, mount it as a volume with `-v` as shown above.

The container has a `HEALTHCHECK` that polls `GET /health` and validates an HTTP 200 response. Docker and Kubernetes readiness/liveness probes can use this automatically.

### Verify the container is healthy

```bash
docker ps                             # check STATUS column shows "(healthy)"
curl http://localhost:8000/health
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov=api --cov-report=term-missing

# Run a specific test file
pytest tests/test_predict.py -v
```

The test suite covers:

| File | Scope |
|---|---|
| `test_data_ingestion.py` | CSV loading, schema validation, empty-file handling |
| `test_preprocessing.py` | Cleaning functions, sklearn pipeline fit/transform/save/load |
| `test_feature_engineering.py` | All 5 derived features, edge cases, immutability checks |
| `test_predict.py` | Inference pipeline, threshold logic, input validation |
| `test_api.py` | All 4 HTTP endpoints with mocked model (no artefacts needed) |

Tests are **mock-based** — they do not require trained artefacts on disk, so they always pass in CI.

---

## Sample API Request / Response

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @samples/predict_single.json
```

**Request body:**

```json
{
  "gender": "Female",
  "seniorcitizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 6,
  "phoneservice": "Yes",
  "multiplelines": "No",
  "internetservice": "Fiber optic",
  "onlinesecurity": "No",
  "onlinebackup": "No",
  "deviceprotection": "No",
  "techsupport": "No",
  "streamingtv": "Yes",
  "streamingmovies": "Yes",
  "contract": "Month-to-month",
  "paperlessbilling": "Yes",
  "paymentmethod": "Electronic check",
  "monthlycharges": 79.85,
  "totalcharges": 479.10
}
```

**Response:**

```json
{
  "churn": 1,
  "probability": 0.8241,
  "label": "Churn",
  "threshold": 0.5
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d @samples/predict_batch.json
```

**Response:**

```json
{
  "scored": 4,
  "predictions": [
    { "churn": 1, "probability": 0.82, "label": "Churn",    "threshold": 0.5 },
    { "churn": 0, "probability": 0.09, "label": "No Churn", "threshold": 0.5 },
    { "churn": 1, "probability": 0.61, "label": "Churn",    "threshold": 0.5 },
    { "churn": 0, "probability": 0.44, "label": "No Churn", "threshold": 0.5 }
  ]
}
```

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

---

## Key Highlights

These are the points worth calling out in a portfolio or interview:

**End-to-end MLOps pipeline**  
Built every layer from scratch: data ingestion → feature engineering → model training → experiment tracking → REST API → containerisation → CI/CD. No AutoML or managed platforms.

**Training-serving skew prevention**  
The `sklearn` `ColumnTransformer` (StandardScaler + OneHotEncoder) is fitted once and serialised as `preprocessor.joblib`. The inference path calls `.transform()` — never `.fit_transform()` — on the same object. This is explicitly tested in `test_predict.py`.

**Configurable decision threshold**  
The API accepts `?threshold=0.35` per request. Retention campaigns can lower the threshold to catch borderline customers without retraining or redeploying the model.

**MLflow parent/child run hierarchy**  
Each call to `train_all_models()` opens a parent `training_session` run. Each individual model is logged as a child run. The best model is automatically registered in the Model Registry as `churn_best_model`.

**Multi-stage Docker build**  
Builder stage installs `requirements-prod.txt` (no dev tools) into an isolated venv. Runtime stage copies only the venv and app code. The `HEALTHCHECK` validates HTTP 200 — not just TCP reachability.

**Mock-based tests — no artefacts needed**  
All tests pass in CI without trained model files. `unittest.mock.patch` injects controlled outputs at the `_run_inference_pipeline` boundary, keeping tests fast (< 2 s for 39 tests) and deterministic.

**GitHub Actions CI with three gates**  
Lint (flake8) → Tests (pytest + coverage threshold 70 %) → Docker build smoke test. The pip cache is keyed on both `requirements.txt` and `requirements-prod.txt` so adding a production dependency correctly busts the cache.

---

## Dataset

**IBM Telco Customer Churn**  
Source: [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
Rows: 7,043 customers · Features: 20 · Target: `Churn` (Yes/No)

> The dataset is not included in this repository. Download it from Kaggle and place it at `data/raw/churn.csv`.

---

## License

MIT — free to use, modify, and distribute with attribution.

