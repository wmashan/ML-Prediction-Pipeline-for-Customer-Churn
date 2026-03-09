"""
predict.py
----------
Inference module for the customer churn prediction pipeline.

Full inference pipeline (mirrors training exactly):
  raw input (dict or DataFrame)
    → _to_dataframe()          convert + type-coerce input into a DataFrame
    → _validate_input()        check required columns; raise on bad input
    → clean()                  Pandas cleaning  (Phase 1 of training)
    → engineer_features()      domain features  (Phase 2 of training)
    → preprocessor.transform() sklearn encode + scale (NEVER re-fit)
    → model.predict_proba()    churn probability
    → structured result dict

Why run clean() and engineer_features() at inference?
─────────────────────────────────────────────────────
The sklearn ColumnTransformer was fitted on a DataFrame that already had
`tenure_group`, `avg_monthly_spend`, `service_count`, `contract_risk_score`,
and `is_digital_only` columns.  If you skip those steps and pass raw data
directly to `.transform()` you get a KeyError — or worse, silently wrong
column alignment → garbage predictions. See the full explanation at the
bottom of this file.

FastAPI integration
───────────────────
This module is intentionally import-friendly:

    from src.predict import predict_single, predict_batch, reload_artefacts

Artefacts are lazy-loaded once and cached at module level so the API
process does NOT reload heavy files on every HTTP request.

ML role   : Translates raw customer data into a churn probability + label.
Data Eng  : Normalises flexible input (dict / DataFrame) to match prod schema.
MLOps     : Decoupled from training; artefact paths are versioned .joblib files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import engineer_features
from src.preprocessing import clean

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models")
_DEFAULT_MODEL_NAME = "best_model"

# ---------------------------------------------------------------------------
# Required raw input columns (before any cleaning or feature engineering).
# These are the minimal columns the ColumnTransformer was trained on.
# Input missing any of these will raise a predictable ValueError rather than
# silently producing wrong predictions.
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: List[str] = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
]

# ---------------------------------------------------------------------------
# Module-level artefact cache
# Loaded once on first prediction request; reused on all subsequent calls.
# Avoids disk I/O on every API request.
# ---------------------------------------------------------------------------

_model: Any = None
_preprocessor: Any = None
_model_name_loaded: str = ""


# ============================================================================
# Artefact loading
# ============================================================================

def _load_artefacts(model_name: str = _DEFAULT_MODEL_NAME) -> None:
    """
    Lazy-load the trained model and fitted preprocessor into module-level
    variables.  Called automatically on the first prediction request.

    Parameters
    ----------
    model_name : Stem of the .joblib file under models/.
                 Default: 'best_model' → models/best_model.joblib

    Raises
    ------
    FileNotFoundError : If either artefact file is missing.
    """
    global _model, _preprocessor, _model_name_loaded

    model_path = MODELS_DIR / f"{model_name}.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artefact not found: {model_path.resolve()}\n"
            "Run `python -m src.train` first to generate it."
        )
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor artefact not found: {preprocessor_path.resolve()}\n"
            "Run `python -m src.train` first to generate it."
        )

    _model = joblib.load(model_path)
    _preprocessor = joblib.load(preprocessor_path)
    _model_name_loaded = model_name

    logger.info(
        "Artefacts loaded — model: %s | preprocessor: %s",
        model_path.resolve(),
        preprocessor_path.resolve(),
    )


def reload_artefacts(model_name: str = _DEFAULT_MODEL_NAME) -> None:
    """
    Force a fresh reload of model + preprocessor from disk.

    Call this after retraining to pick up the new artefacts without
    restarting the API process.

    Parameters
    ----------
    model_name : Model file stem (same as _load_artefacts).
    """
    global _model, _preprocessor, _model_name_loaded
    _model = None
    _preprocessor = None
    _model_name_loaded = ""
    _load_artefacts(model_name)
    logger.info("Artefacts reloaded successfully.")


def get_artefact_info() -> Dict[str, Any]:
    """
    Return metadata about the currently loaded artefacts.

    Useful as a health-check payload in the FastAPI /health endpoint.

    Returns
    -------
    dict with:
        model_loaded        : bool
        preprocessor_loaded : bool
        model_name          : str  (name used to load the model)
        model_type          : str  (class name of the fitted estimator)
    """
    return {
        "model_loaded":        _model is not None,
        "preprocessor_loaded": _preprocessor is not None,
        "model_name":          _model_name_loaded or "not loaded",
        "model_type":          type(_model).__name__ if _model is not None else "n/a",
    }


# ============================================================================
# Input helpers
# ============================================================================

def _to_dataframe(input_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """
    Normalise raw input into a single-row or multi-row pandas DataFrame.

    Accepted input types
    --------------------
    dict       : One customer record  → converted to a single-row DataFrame.
    DataFrame  : One or more customers → used as-is (deep copy so we don't
                 mutate the caller's object).

    Column names are lowercased and stripped so callers can pass either
    'TotalCharges' or 'totalcharges' without hitting a KeyError later.

    Parameters
    ----------
    input_data : Customer feature data.

    Returns
    -------
    pd.DataFrame  (always at least one row; column names lowercased).

    Raises
    ------
    TypeError  : If input_data is not a dict or DataFrame.
    ValueError : If input_data is an empty dict or empty DataFrame.
    """
    if isinstance(input_data, dict):
        if not input_data:
            raise ValueError("Input dict is empty. Provide at least one field.")
        df = pd.DataFrame([input_data])

    elif isinstance(input_data, pd.DataFrame):
        if input_data.empty:
            raise ValueError("Input DataFrame is empty. Provide at least one row.")
        df = input_data.copy()

    else:
        raise TypeError(
            f"input_data must be a dict or pd.DataFrame, "
            f"got {type(input_data).__name__}."
        )

    # Normalise column names: lowercase + strip whitespace.
    # Mirrors what preprocessing.standardise_columns() does so that
    # _validate_input() can reliably check against REQUIRED_COLUMNS.
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )

    return df


def _validate_input(df: pd.DataFrame) -> None:
    """
    Check that all required columns are present in the input DataFrame.

    Does NOT validate value ranges or types — those are handled downstream
    by clean() and the sklearn pipeline's internal SimpleImputers.

    Parameters
    ----------
    df : Normalised DataFrame (column names already lowercased).

    Raises
    ------
    ValueError : lists every missing column so the caller knows exactly
                 what to fix, rather than getting a cryptic KeyError later.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required column(s):\n"
            f"  {missing}\n"
            f"Received columns: {df.columns.tolist()}"
        )


# ============================================================================
# Core inference pipeline
# ============================================================================

def _run_inference_pipeline(
    df: pd.DataFrame,
    model_name: str = _DEFAULT_MODEL_NAME,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the full inference pipeline on a pre-validated DataFrame.

    Steps
    -----
    1. clean()               — same Pandas cleaning as training Phase 1
    2. engineer_features()   — same domain features as training Phase 2
    3. preprocessor.transform() — apply fitted sklearn ColumnTransformer
    4. model.predict_proba() — return raw probability array

    NOTE: The `churn` column is absent from inference input.
    clean() handles this gracefully (skips target encoding if absent).

    Parameters
    ----------
    df         : Validated raw DataFrame (output of _to_dataframe + _validate_input).
    model_name : Model file stem to use.

    Returns
    -------
    probs     : shape (n_samples,)  — churn probability for each row
    X_encoded : shape (n_samples, n_features) — encoded feature matrix
    """
    # Ensure artefacts are in memory
    if _model is None or _preprocessor is None:
        _load_artefacts(model_name)

    # ── Phase 1: Pandas cleaning (mirrors training) ────────────────────────
    # Standardises column names, fixes dtypes, fills nulls.
    # target_col='churn' is safe even when 'churn' is absent —
    # clean() skips the target encoding step if the column doesn't exist.
    df_clean = clean(df, target_col="churn")

    # ── Phase 2: Feature engineering (mirrors training) ────────────────────
    # Adds tenure_group, avg_monthly_spend, service_count,
    # contract_risk_score, is_digital_only.
    # The ColumnTransformer expects these columns — skip this and
    # .transform() raises a KeyError.
    df_engineered = engineer_features(df_clean)

    # ── Phase 3: Sklearn transform (NEVER re-fit) ───────────────────────────
    # Drop the target column if somehow present (e.g. caller sent a labelled
    # row); the ColumnTransformer was fitted without it.
    feature_df = df_engineered.drop(columns=["churn"], errors="ignore")
    X_encoded: np.ndarray = _preprocessor.transform(feature_df)

    # ── Phase 4: Predict ───────────────────────────────────────────────────
    probs: np.ndarray = _model.predict_proba(X_encoded)[:, 1]  # P(churn=1)

    return probs, X_encoded


# ============================================================================
# Public API — used by FastAPI and batch scripts
# ============================================================================

def predict_single(
    input_data: Union[Dict[str, Any], pd.DataFrame],
    model_name: str = _DEFAULT_MODEL_NAME,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Predict churn for a single customer record.

    Parameters
    ----------
    input_data : Raw customer features as a dict or single-row DataFrame.
                 Column names are case-insensitive (normalised internally).
    model_name : Model file stem to load from models/.
    threshold  : Probability cutoff above which a customer is labelled "Churn".
                 Default 0.5.  Lower this (e.g. 0.35) to catch more churners
                 at the cost of more false positives.

    Returns
    -------
    dict with keys:
        churn       : int   — 1 (will churn) or 0 (will stay)
        probability : float — probability of churn, rounded to 4 d.p.
        label       : str   — "Churn" or "No Churn"
        threshold   : float — the cutoff used for this prediction

    Raises
    ------
    TypeError        : input_data is not a dict or DataFrame.
    ValueError       : required columns are missing or input is empty.
    FileNotFoundError: model/preprocessor .joblib files not found.

    Example
    -------
    >>> result = predict_single({
    ...     "gender": "Female", "seniorcitizen": 0, "partner": "Yes",
    ...     "dependents": "No", "tenure": 12, "phoneservice": "Yes",
    ...     "multiplelines": "No", "internetservice": "Fiber optic",
    ...     "onlinesecurity": "No", "onlinebackup": "No",
    ...     "deviceprotection": "No", "techsupport": "No",
    ...     "streamingtv": "Yes", "streamingmovies": "Yes",
    ...     "contract": "Month-to-month", "paperlessbilling": "Yes",
    ...     "paymentmethod": "Electronic check",
    ...     "monthlycharges": 79.85, "totalcharges": 958.2,
    ... })
    >>> result
    {'churn': 1, 'probability': 0.7431, 'label': 'Churn', 'threshold': 0.5}
    """
    df = _to_dataframe(input_data)
    _validate_input(df)

    probs, _ = _run_inference_pipeline(df, model_name=model_name)
    prob = float(probs[0])
    label_int = int(prob >= threshold)

    result: Dict[str, Any] = {
        "churn":       label_int,
        "probability": round(prob, 4),
        "label":       "Churn" if label_int == 1 else "No Churn",
        "threshold":   threshold,
    }

    logger.info(
        "predict_single → %s (prob=%.4f, threshold=%.2f)",
        result["label"], prob, threshold,
    )
    return result


def predict_batch(
    input_data: Union[List[Dict[str, Any]], pd.DataFrame],
    model_name: str = _DEFAULT_MODEL_NAME,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Predict churn for multiple customers in one call.

    Parameters
    ----------
    input_data : Either a list of customer dicts or a multi-row DataFrame.
    model_name : Model file stem.
    threshold  : Probability cutoff for the churn label.

    Returns
    -------
    pd.DataFrame — the original rows with three new columns appended:
        churn_probability : float  — P(churn=1) for each row
        churn_label       : str    — "Churn" or "No Churn"
        churn_prediction  : int    — 1 or 0

    Raises
    ------
    TypeError        : input_data is not a list or DataFrame.
    ValueError       : required columns are missing or input is empty.
    FileNotFoundError: model/preprocessor .joblib files not found.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.read_csv("data/raw/churn.csv")
    >>> scored = predict_batch(df.drop(columns=["Churn"]))
    >>> scored[["churn_probability", "churn_label"]].head()
    """
    # Convert list-of-dicts to DataFrame
    if isinstance(input_data, list):
        if not input_data:
            raise ValueError("Input list is empty.")
        df = _to_dataframe(pd.DataFrame(input_data))
    else:
        df = _to_dataframe(input_data)

    _validate_input(df)

    probs, _ = _run_inference_pipeline(df, model_name=model_name)

    # Append predictions to a copy of the normalised input
    result_df = df.copy()
    result_df["churn_probability"] = np.round(probs, 4)
    result_df["churn_prediction"]  = (probs >= threshold).astype(int)
    result_df["churn_label"]       = np.where(probs >= threshold, "Churn", "No Churn")

    churners = int((probs >= threshold).sum())
    logger.info(
        "predict_batch → %d rows scored | %d predicted to churn (%.1f%%)",
        len(df), churners, churners / len(df) * 100,
    )
    return result_df


# ============================================================================
# Standalone smoke-test
# Run:  python -m src.predict
# Requires models/ artefacts to be present (run src/train.py first).
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    sample_customer: Dict[str, Any] = {
        "gender":          "Female",
        "seniorcitizen":   0,
        "partner":         "Yes",
        "dependents":      "No",
        "tenure":          12,
        "phoneservice":    "Yes",
        "multiplelines":   "No",
        "internetservice": "Fiber optic",
        "onlinesecurity":  "No",
        "onlinebackup":    "No",
        "deviceprotection":"No",
        "techsupport":     "No",
        "streamingtv":     "Yes",
        "streamingmovies": "Yes",
        "contract":        "Month-to-month",
        "paperlessbilling":"Yes",
        "paymentmethod":   "Electronic check",
        "monthlycharges":  79.85,
        "totalcharges":    958.20,
    }

    try:
        result = predict_single(sample_customer)
        print("\nSingle-customer prediction:")
        for k, v in result.items():
            print(f"  {k:<14}: {v}")

        print("\nArtefact info:")
        for k, v in get_artefact_info().items():
            print(f"  {k:<22}: {v}")

    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("Hint: train the model first →  python -m src.train")


# ============================================================================
# Why training and inference must use the SAME preprocessing pipeline
# ============================================================================
#
# During training, the sklearn ColumnTransformer LEARNS statistics from the
# training data:
#   • StandardScaler  records the mean and std of every numeric column.
#   • OneHotEncoder   records the full category list for every categorical column.
#
# At inference time, .transform() applies those EXACT same statistics to new data.
#
# If you re-fit the pipeline on production data instead, you introduce
# "training/serving skew" — the numeric scales and category sets differ,
# so the model receives inputs that look nothing like the data it was trained on.
#
# Concrete example — StandardScaler on `tenure`:
#   Training mean = 32.4 months.  Scaler produces:  z = (tenure - 32.4) / std
#   If you re-fit on a single incoming row, mean = that row's tenure, z = 0
#   for every row → the model sees a completely different signal.
#
# Concrete example — OneHotEncoder on `contract`:
#   Training encoder saw: month-to-month, one year, two year → 2-column OHE.
#   If you re-fit on a batch that only contains "month-to-month", it outputs
#   a 0-column matrix → shape mismatch → crash or silent wrong features.
#
# Concrete example — missing engineered columns:
#   The ColumnTransformer was fitted AFTER engineer_features() ran, so it
#   knows about `tenure_group` and `service_count`.  Skip engineer_features()
#   at inference and .transform() raises KeyError immediately.
#
# The fix (implemented in _run_inference_pipeline):
#   1. Fit the ColumnTransformer ONCE on training data in train.py.
#   2. Save it to models/preprocessor.joblib with joblib.dump().
#   3. At inference: load with joblib.load(), call .transform() — never .fit().
#   4. Run clean() and engineer_features() first so all derived columns exist.
#
# This pattern is the "feature pipeline contract" between training and serving:
# the saved pipeline object IS the versioned agreement on what the model expects.

