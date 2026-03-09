"""
train.py
--------
Trains multiple ML models, tracks every experiment with MLflow,
and saves the best model + preprocessor to disk.

Responsibilities:
  1. Accept engineered features.
  2. Train a configurable set of classifiers.
  3. Log params, metrics, and artefacts to MLflow.
  4. Select the best model by ROC-AUC.
  5. Persist model + preprocessor using joblib.

ML role       : Core training loop with model selection.
Data Eng role : Consumes the processed data produced upstream.
MLOps role    : MLflow tracking + model registry + artefact storage = full
                experiment reproducibility.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalogue – add or remove models here without touching training logic
# ---------------------------------------------------------------------------
MODEL_CATALOGUE: Dict[str, Any] = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    ),
}

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Single-model training + MLflow logging
# ---------------------------------------------------------------------------

def train_single_model(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Fit one model, log it to MLflow, return its ROC-AUC on the test set.

    Parameters
    ----------
    name    : Model identifier (used as MLflow run name).
    model   : Un-fitted sklearn estimator.
    X_train, y_train : Training data.
    X_test,  y_test  : Evaluation data.

    Returns
    -------
    float : ROC-AUC score on the test set.
    """
    with mlflow.start_run(run_name=name):
        # ------------------------------------------------------------------
        # 1. Log hyperparameters
        # ------------------------------------------------------------------
        mlflow.log_params(model.get_params())

        # ------------------------------------------------------------------
        # 2. Train
        # ------------------------------------------------------------------
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)

        # ------------------------------------------------------------------
        # 3. Evaluate
        # ------------------------------------------------------------------
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("roc_auc", roc_auc)
        logger.info("  %s → ROC-AUC: %.4f", name, roc_auc)

        # ------------------------------------------------------------------
        # 4. Log model artefact to MLflow
        # ------------------------------------------------------------------
        mlflow.sklearn.log_model(model, artifact_path=name)

    return roc_auc


# ---------------------------------------------------------------------------
# Multi-model training loop
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str = "churn_prediction",
    tracking_uri: str = "mlruns",
) -> Tuple[str, Any, float]:
    """
    Train all models in MODEL_CATALOGUE, track with MLflow, and return the best.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : Feature arrays from feature_engineering.
    experiment_name : MLflow experiment name.
    tracking_uri    : Local directory or remote MLflow server URI.

    Returns
    -------
    best_name  : Name of the winning model.
    best_model : Fitted sklearn estimator.
    best_auc   : Its ROC-AUC score.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    results: Dict[str, Tuple[Any, float]] = {}

    for name, model in MODEL_CATALOGUE.items():
        auc = train_single_model(name, model, X_train, y_train, X_test, y_test)
        results[name] = (model, auc)

    # Pick best by ROC-AUC
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_auc = results[best_name]

    logger.info("Best model: %s (ROC-AUC=%.4f)", best_name, best_auc)
    return best_name, best_model, best_auc


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: Any, preprocessor: Any, model_name: str = "best_model") -> None:
    """
    Persist the trained model and its preprocessor to the models/ directory.

    Both must be saved together so inference can apply identical transforms.
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info("Model saved       → %s", model_path)
    logger.info("Preprocessor saved → %s", preprocessor_path)


def load_model(model_name: str = "best_model") -> Tuple[Any, Any]:
    """
    Load a previously saved model and preprocessor from disk.

    Returns
    -------
    model, preprocessor
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Loaded model from %s", model_path)
    return model, preprocessor
