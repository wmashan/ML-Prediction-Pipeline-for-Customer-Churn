"""
predict.py
----------
Loads a saved model + preprocessor and runs inference on new data.

This module is used by:
  - The FastAPI service (api/main.py) for online single-row predictions.
  - Batch scripts for offline scoring of large files.

Design principle: The preprocessor MUST be loaded here. Never re-fit it on
new data – that would cause training/serving skew (a classic MLOps bug).

ML role       : Translates raw input into a churn probability + label.
Data Eng role : Accepts raw dict/DataFrame input, mirrors prod schema.
MLOps role    : Decoupled from training; model artefacts are versioned files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Loader (cached at module level so the API doesn't reload on every request)
# ---------------------------------------------------------------------------

_model = None
_preprocessor = None


def _load_artefacts(model_name: str = "best_model") -> None:
    """
    Lazy-load model and preprocessor into module-level variables.
    Called automatically on first prediction request.
    """
    global _model, _preprocessor

    model_path = MODELS_DIR / f"{model_name}.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path.resolve()}")

    _model = joblib.load(model_path)
    _preprocessor = joblib.load(preprocessor_path)
    logger.info("Artefacts loaded from %s", MODELS_DIR.resolve())


# ---------------------------------------------------------------------------
# Core prediction functions
# ---------------------------------------------------------------------------

def predict_single(
    input_data: Dict[str, Any],
    model_name: str = "best_model",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Predict churn for a single customer record.

    Parameters
    ----------
    input_data : Raw feature dict (matches training schema).
    threshold  : Probability cutoff to label as churn.

    Returns
    -------
    dict with keys:
      - 'churn'       : 1 (will churn) or 0 (will stay)
      - 'probability' : float, probability of churn
      - 'label'       : 'Churn' or 'No Churn'
    """
    if _model is None or _preprocessor is None:
        _load_artefacts(model_name)

    df = pd.DataFrame([input_data])
    X = _preprocessor.transform(df)
    prob = float(_model.predict_proba(X)[0, 1])
    label_int = int(prob >= threshold)

    result = {
        "churn": label_int,
        "probability": round(prob, 4),
        "label": "Churn" if label_int == 1 else "No Churn",
    }

    logger.info("Prediction → %s (prob=%.4f)", result["label"], prob)
    return result


def predict_batch(
    input_df: pd.DataFrame,
    model_name: str = "best_model",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Predict churn for a DataFrame of customers (batch inference).

    Parameters
    ----------
    input_df   : Raw feature DataFrame.
    threshold  : Probability cutoff.

    Returns
    -------
    input_df with two extra columns: 'churn_probability', 'churn_label'.
    """
    if _model is None or _preprocessor is None:
        _load_artefacts(model_name)

    df = input_df.copy()
    X = _preprocessor.transform(df)
    probs = _model.predict_proba(X)[:, 1]

    df["churn_probability"] = np.round(probs, 4)
    df["churn_label"] = np.where(probs >= threshold, "Churn", "No Churn")

    logger.info("Batch prediction complete. %d rows scored.", len(df))
    return df
