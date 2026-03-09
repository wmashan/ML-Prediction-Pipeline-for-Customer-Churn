# Customer Churn ML Prediction Pipeline

A production-style, end-to-end machine learning system that predicts whether a telecom customer will churn.  
Built step-by-step as a learning project covering **AI/ML Engineering**, **Data Engineering**, and **MLOps**.

---

## Project Structure

```
customer-churn-ml-pipeline/
│
├── data/
│   ├── raw/              ← Drop raw CSV here (e.g., churn.csv from Kaggle)
│   └── processed/        ← Cleaned & feature-engineered data (auto-generated)
│
├── notebooks/            ← Exploratory Data Analysis (Jupyter notebooks)
│
├── src/
│   ├── data_ingestion.py     ← Load raw CSV → DataFrame
│   ├── preprocessing.py      ← Clean data (nulls, types, duplicates)
│   ├── feature_engineering.py← Encode, scale, derive new features
│   ├── train.py              ← Train models, track with MLflow, save best
│   ├── evaluate.py           ← Metrics, confusion matrix, ROC curve
│   ├── predict.py            ← Load saved model, run inference
│   └── utils.py              ← Logging, config, timer, save helpers
│
├── models/               ← Saved model + preprocessor (joblib files)
│
├── api/
│   └── main.py           ← FastAPI REST service (predict / batch / health)
│
├── tests/
│   ├── test_preprocessing.py ← Unit tests for cleaning logic
│   └── test_api.py           ← Integration tests for API endpoints
│
├── .github/
│   └── workflows/
│       └── ci.yml        ← GitHub Actions: lint → test → docker build
│
├── Dockerfile            ← Multi-stage Docker build
├── requirements.txt      ← Pinned Python dependencies
├── README.md             ← You are here
└── .gitignore
```

---

## Quick Start

### 1. Get the dataset

Download the **Telco Customer Churn** dataset from Kaggle:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the file at: `data/raw/churn.csv`

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the full training pipeline

```bash
# From the project root
python -c "
from src.utils import setup_logging, load_config, timer
from src.data_ingestion import load_csv
from src.preprocessing import preprocess
from src.feature_engineering import engineer_features
from src.train import train_all_models, save_model
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve

setup_logging()
cfg = load_config()

with timer('Full pipeline'):
    df = load_csv(cfg['data_path'])
    df = preprocess(df)
    X_train, X_test, y_train, y_test, preprocessor = engineer_features(df)
    best_name, best_model, best_auc = train_all_models(X_train, y_train, X_test, y_test)
    save_model(best_model, preprocessor)
    metrics = evaluate_model(best_model, X_test, y_test, model_name=best_name)
    plot_confusion_matrix(best_model, X_test, y_test)
    plot_roc_curve(best_model, X_test, y_test)
    print('Best model:', best_name, '| ROC-AUC:', round(best_auc, 4))
"
```

### 4. View MLflow experiments

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 5. Run the API locally

```bash
uvicorn api.main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

### 6. Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-api
```

### 7. Run tests

```bash
pytest tests/ -v --cov=src --cov=api
```

---

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Liveness probe                       |
| GET    | `/model-info`     | Returns current model name           |
| POST   | `/predict`        | Single customer churn prediction     |
| POST   | `/predict-batch`  | Batch scoring for multiple customers |

Interactive docs: `http://localhost:8000/docs`

---

## Models Trained

| Model                | Notes                           |
|----------------------|---------------------------------|
| Logistic Regression  | Fast baseline, interpretable    |
| Random Forest        | Ensemble, handles non-linearity |
| Gradient Boosting    | Often best performance          |

Best model is selected by **ROC-AUC** on the held-out test set.

---

## Tech Stack

| Area            | Tools                                     |
|-----------------|-------------------------------------------|
| ML / Data       | Python, Pandas, NumPy, Scikit-learn       |
| Experiment Tracking | MLflow                               |
| API             | FastAPI, Uvicorn, Pydantic                |
| Containerisation | Docker (multi-stage)                    |
| CI/CD           | GitHub Actions                            |
| Testing         | Pytest, pytest-cov                        |
| Code Quality    | Flake8                                    |

---

## Step-by-Step Guide (for learners)

Each file in `src/` does exactly one thing:

1. **data_ingestion.py** – Load CSV → raw DataFrame  
2. **preprocessing.py** – Clean data (nulls, types, duplicates)  
3. **feature_engineering.py** – Encode categories, scale numerics, create new features  
4. **train.py** – Train multiple models, track experiments with MLflow  
5. **evaluate.py** – Compute metrics, generate plots  
6. **predict.py** – Load saved model, run inference  
7. **utils.py** – Shared helpers (logging, config, timer)

This separation means each part can be tested, replaced, or deployed independently.

---

## Interview Topics Covered

- Handling class imbalance (class_weight="balanced")
- Preventing data leakage (fit scaler on train only)
- Training/serving skew (save and reuse the preprocessor)
- Model versioning with MLflow
- Quality gates in CI (min ROC-AUC threshold)
- Multi-stage Docker builds (smaller, more secure images)
- REST API design with FastAPI + Pydantic validation

---

## Next Steps

- [ ] Add a Streamlit dashboard (`streamlit run app.py`)
- [ ] Connect to PostgreSQL for production data
- [ ] Add SHAP explainability
- [ ] Deploy to AWS / GCP / Azure
- [ ] Add CD pipeline for automatic model deployment
