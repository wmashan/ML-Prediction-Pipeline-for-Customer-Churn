"""
preprocessing.py
----------------
Cleans raw data AND builds a reusable sklearn transformation pipeline
that encodes categorical features and scales numerical features.

Two-phase design
────────────────
Phase 1 — Pandas cleaning  (functions: clean())
  Operates on a raw DataFrame. Fixes types, removes duplicates, fills nulls,
  standardises column names. Returns a clean DataFrame.

Phase 2 — Sklearn Pipeline  (functions: build_pipeline(), fit_pipeline(), apply_pipeline())
  Takes the clean DataFrame, fits an sklearn ColumnTransformer that:
    • Scales numeric columns    → StandardScaler
    • Encodes categorical cols  → OneHotEncoder
  Returns a numpy array ready for model training or inference.

Why sklearn Pipeline + ColumnTransformer?
────────────────────────────────────────
1. NO DATA LEAKAGE
   fit() is called only on training data. transform() is called on test/prod data.
   If you fit on all data, test-set statistics leak into training → inflated metrics.

2. TRAINING / SERVING CONSISTENCY
   Save the fitted pipeline with joblib. At inference time load it and call
   .transform() — the exact same scaler mean/std and encoder categories are used.
   Fitting a new scaler on production data each time causes training/serving skew.

3. SINGLE OBJECT TO VERSION
   One joblib file contains ALL transforms. You don't need to save 5 separate scalers.

4. ATOMIC COLUMNS
   ColumnTransformer applies the right transform to the right columns automatically.
   Adding a new feature only requires updating the column lists — not rewriting logic.

ML role       : Produces model-ready feature matrices with zero manual loops.
Data Eng role : Data quality layer + transform layer; raw → processed zone.
MLOps role    : Serialisable pipeline → reproducible predictions in production.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Default path where the fitted pipeline is saved / loaded
PIPELINE_SAVE_PATH = Path("models/preprocessor.joblib")

# Columns to always drop before building the feature matrix.
# customerid is a unique identifier — not a feature.
DROP_COLUMNS: List[str] = ["customerid"]


# ============================================================================
# PHASE 1 — Pandas Cleaning Helpers
# ============================================================================

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase all column names and replace spaces/hyphens with underscores.

    Why: consistent column names mean every downstream function can use
    string literals without worrying about casing or spaces.

    Example: 'Total Charges' → 'total_charges'
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    logger.debug("Columns after standardisation: %s", df.columns.tolist())
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows and log how many were removed.

    Why: duplicate rows inflate training data and can bias the model
    toward seeing certain patterns more often than they actually occur.
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.warning("Dropped %d duplicate rows.", removed)
    return df


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Fill or drop missing values before they reach the sklearn pipeline.

    Why handle nulls here (in Pandas) rather than leaving it to the sklearn
    SimpleImputer inside the Pipeline?
      - The sklearn imputer handles nulls in the numeric/categorical columns
        it knows about. But Pandas-level cleaning catches structural problems
        first: e.g. TotalCharges stored as a string with whitespace (" ") —
        that becomes NaN only after a pd.to_numeric() cast, which happens before
        the sklearn pipeline ever runs.
      - We also drop entire columns here if > `drop_threshold` fraction is null.
        The sklearn imputer can't drop a column it's been told to transform.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_strategy : {'median', 'mean', 'zero'}
        How to fill missing numeric columns.
    categorical_strategy : {'most_frequent', 'unknown'}
        How to fill missing categorical columns.
    drop_threshold : float
        Drop any column where more than this fraction of values are missing.
    """
    df = df.copy()

    # Drop columns that are mostly empty
    missing_frac = df.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > drop_threshold].index.tolist()
    if cols_to_drop:
        logger.warning("Dropping high-missing columns (>%.0f%%): %s", drop_threshold * 100, cols_to_drop)
        df.drop(columns=cols_to_drop, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols     = df.select_dtypes(include=["object", "category"]).columns

    # Fill numeric nulls
    for col in numeric_cols:
        if df[col].isnull().any():
            if numeric_strategy == "median":
                fill_val = df[col].median()
            elif numeric_strategy == "mean":
                fill_val = df[col].mean()
            else:
                fill_val = 0
            df[col] = df[col].fillna(fill_val)
            logger.debug("Filled numeric '%s' with %s=%.4f", col, numeric_strategy, fill_val)

    # Fill categorical nulls
    for col in cat_cols:
        if df[col].isnull().any():
            fill_val = (
                df[col].mode()[0]
                if categorical_strategy == "most_frequent"
                else "unknown"
            )
            df[col] = df[col].fillna(fill_val)
            logger.debug("Filled categorical '%s' with '%s'", col, fill_val)

    return df


def cast_types(df: pd.DataFrame, target_col: str = "churn") -> pd.DataFrame:
    """
    Fix columns that have the wrong Python/Pandas dtype.

    Why this is a separate step:
      Pandas dtype bugs (e.g. a numeric column stored as object) will cause
      sklearn to raise a ValueError later. It's better to surface them here
      with a clear operation than to get a cryptic sklearn error.

    Transformations:
      - 'totalcharges' : Telco dataset stores this as a string with whitespace
                         entries (" "). pd.to_numeric with errors='coerce' turns
                         those into NaN, which is then filled by handle_missing_values().
      - target column  : Mapped from 'Yes'/'No' → 1/0 integer.
    """
    df = df.copy()

    # Telco-specific: TotalCharges arrives as object with whitespace rows
    if "totalcharges" in df.columns:
        df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
        logger.debug("Cast 'totalcharges' to numeric.")

    # Encode binary target label: 'Yes' → 1, 'No' → 0
    if target_col in df.columns:
        if df[target_col].dtype == object:
            df[target_col] = (
                df[target_col].str.strip().str.lower()
                .map({"yes": 1, "no": 0})
            )
        df[target_col] = df[target_col].astype(int)
        logger.debug("Encoded target '%s' as 0/1 integer.", target_col)

    return df


# ============================================================================
# PHASE 1 ORCHESTRATOR — Pandas Cleaning
# ============================================================================

def clean(
    df: pd.DataFrame,
    target_col: str = "churn",
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run all Pandas-level cleaning steps in order.

    This function only cleans — it does NOT encode or scale.
    Encoding and scaling are done by the sklearn pipeline in Phase 2.

    Steps (in order):
      1. Standardise column names   (lowercase, underscores)
      2. Remove duplicate rows
      3. Fix data types             (TotalCharges → float, Churn → 0/1)
      4. Fill missing values        (median for numeric, mode for categorical)
      5. Drop identifier columns    (e.g. customerid)

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from data_ingestion.ingest().
    target_col : str
        Name of the target column (after standardisation → lowercase).
    drop_cols : list[str] | None
        Extra columns to drop. Defaults to DROP_COLUMNS (customerid).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with:
          - Standardised column names
          - No duplicates
          - No missing values
          - Correct dtypes
          - Identifier columns removed
    """
    logger.info("Phase 1 — Cleaning. Input shape: %s", df.shape)

    df = standardise_columns(df)
    df = drop_duplicates(df)
    df = cast_types(df, target_col=target_col)
    df = handle_missing_values(df)

    # Drop non-feature identifier columns
    cols_to_drop = drop_cols if drop_cols is not None else DROP_COLUMNS
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)
        logger.debug("Dropped identifier columns: %s", existing_drops)

    logger.info("Phase 1 complete. Output shape: %s", df.shape)
    return df


# ============================================================================
# PHASE 2 — Sklearn Pipeline (Encoding + Scaling)
# ============================================================================

def get_column_types(
    df: pd.DataFrame,
    target_col: str = "churn",
) -> Tuple[List[str], List[str]]:
    """
    Automatically detect numeric and categorical feature columns.

    Excludes the target column from both lists so it never gets transformed.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of clean()).
    target_col : str
        Column to exclude (it's a label, not a feature).

    Returns
    -------
    numeric_cols : list[str]   columns with int/float dtype
    cat_cols     : list[str]   columns with object/category dtype
    """
    feature_df = df.drop(columns=[target_col], errors="ignore")

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info("Numeric features  (%d): %s", len(numeric_cols), numeric_cols)
    logger.info("Categorical features (%d): %s", len(cat_cols), cat_cols)

    return numeric_cols, cat_cols


def build_pipeline(
    numeric_cols: List[str],
    cat_cols: List[str],
) -> ColumnTransformer:
    """
    Build (but do NOT fit) the sklearn ColumnTransformer pipeline.

    Structure
    ---------
    ColumnTransformer
    ├── numeric_pipeline   → applied to numeric_cols
    │     ├── SimpleImputer(strategy='median')   fills any remaining NaNs
    │     └── StandardScaler()                   zero mean, unit variance
    │
    └── categorical_pipeline → applied to cat_cols
          ├── SimpleImputer(strategy='most_frequent')  fills any remaining NaNs
          └── OneHotEncoder(drop='first', handle_unknown='ignore')
                  drop='first'          avoids the dummy variable trap
                  handle_unknown='ignore' won't crash if prod data has
                                          a category not seen in training

    Why two SimpleImputers inside the pipeline?
      Even after Pandas cleaning, a few NaNs can appear at inference time
      if the incoming request has a missing field. The imputers inside the
      pipeline are a safety net for production traffic.

    Parameters
    ----------
    numeric_cols : list[str]   Feature column names that are numeric.
    cat_cols     : list[str]   Feature column names that are categorical.

    Returns
    -------
    ColumnTransformer (unfitted — call .fit_transform() or .fit() + .transform())
    """
    # ── Numeric sub-pipeline ───────────────────────────────────────────────
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # ── Categorical sub-pipeline ───────────────────────────────────────────
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",             # remove one category to avoid multicollinearity
            handle_unknown="ignore",  # unknown categories at inference → all zeros
            sparse_output=False,      # return dense array (easier to work with)
        )),
    ])

    # ── Combine with ColumnTransformer ─────────────────────────────────────
    # remainder='drop' means any column not in numeric_cols or cat_cols is
    # silently ignored (e.g. if a new column appears in production data).
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,      numeric_cols),
            ("cat", categorical_pipeline,  cat_cols),
        ],
        remainder="drop",
    )

    logger.debug(
        "Pipeline built: %d numeric cols + %d categorical cols.",
        len(numeric_cols), len(cat_cols),
    )
    return preprocessor


def fit_pipeline(
    df: pd.DataFrame,
    target_col: str = "churn",
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    """
    Fit the ColumnTransformer on training data and return transformed arrays.

    IMPORTANT: Call this ONLY on training data.
    Calling this on test data leaks statistics (mean, std, categories)
    from the test set into the transformers → inflated evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned training DataFrame (from clean()).
    target_col : str
        Name of the target column.

    Returns
    -------
    pipeline   : Fitted ColumnTransformer — save this with save_pipeline().
    X_train    : Transformed feature matrix (numpy array).
    y_train    : Target array (numpy array of 0/1).
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col].values

    numeric_cols, cat_cols = get_column_types(df, target_col=target_col)
    pipeline = build_pipeline(numeric_cols, cat_cols)

    logger.info("Fitting ColumnTransformer on %d training rows...", len(X))
    X_transformed = pipeline.fit_transform(X)  # ← fit + transform in one step

    logger.info(
        "Pipeline fitted. Output shape: %s  |  Feature matrix dtype: %s",
        X_transformed.shape, X_transformed.dtype,
    )
    return pipeline, X_transformed, y


def apply_pipeline(
    pipeline: ColumnTransformer,
    df: pd.DataFrame,
    target_col: str = "churn",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply an already-fitted pipeline to new data (test set or production).

    Uses .transform() — NOT .fit_transform().
    This is the critical distinction between training and inference:
      Training : pipeline.fit_transform(X_train)  ← learns statistics
      Inference: pipeline.transform(X_new)         ← applies learned statistics

    Parameters
    ----------
    pipeline   : Fitted ColumnTransformer (from fit_pipeline or load_pipeline).
    df         : New data DataFrame (cleaned, same schema as training data).
    target_col : If present in df, it will be extracted and returned.
                 If absent (real production data), y is returned as None.

    Returns
    -------
    X_transformed : numpy array of encoded/scaled features.
    y             : numpy array of labels if target_col exists, else None.
    """
    y: Optional[np.ndarray] = None

    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col].values
    else:
        X = df

    X_transformed = pipeline.transform(X)  # ← ONLY transform, never re-fit

    logger.info("Pipeline applied. Output shape: %s", X_transformed.shape)
    return X_transformed, y


# ============================================================================
# Pipeline persistence — save & load with joblib
# ============================================================================

def save_pipeline(
    pipeline: ColumnTransformer,
    path: str | Path = PIPELINE_SAVE_PATH,
) -> None:
    """
    Serialise a fitted ColumnTransformer to disk.

    Why joblib instead of pickle?
      joblib is optimised for large numpy arrays (which sklearn objects contain).
      It's faster and produces smaller files than standard pickle.

    Parameters
    ----------
    pipeline : Fitted ColumnTransformer.
    path     : Destination .joblib file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Preprocessor pipeline saved → %s", path.resolve())


def load_pipeline(
    path: str | Path = PIPELINE_SAVE_PATH,
) -> ColumnTransformer:
    """
    Load a previously fitted ColumnTransformer from disk.

    Use this in:
      - predict.py   (inference / FastAPI)
      - evaluate.py  (to transform the test set using TRAINING statistics)

    Parameters
    ----------
    path : Path to the saved .joblib file.

    Returns
    -------
    ColumnTransformer (already fitted — call .transform() directly).

    Raises
    ------
    FileNotFoundError if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor pipeline not found at: {path.resolve()}\n"
            "Run the training pipeline first to generate this file."
        )
    pipeline = joblib.load(path)
    logger.info("Preprocessor pipeline loaded from %s", path.resolve())
    return pipeline


# ============================================================================
# Convenience: Full pipeline run in one call (used in train.py)
# ============================================================================

def run_preprocessing(
    df: pd.DataFrame,
    target_col: str = "churn",
    save_path: Optional[str | Path] = PIPELINE_SAVE_PATH,
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    """
    Full Phase 1 + Phase 2 preprocessing in a single function call.

    Intended for use in train.py:

        from src.preprocessing import run_preprocessing

        pipeline, X_train, y_train = run_preprocessing(df_train)

    Steps:
      1. clean()        — Pandas cleaning
      2. fit_pipeline() — sklearn encode + scale (fit on this data)
      3. save_pipeline()— persist fitted pipeline to disk

    Parameters
    ----------
    df         : Raw DataFrame from data_ingestion.ingest().
    target_col : Target column name (after standardisation → lowercase).
    save_path  : Where to save the fitted pipeline. Pass None to skip saving.

    Returns
    -------
    pipeline, X_train, y_train
    """
    # Phase 1: Clean
    df_clean = clean(df, target_col=target_col)

    # Phase 2: Fit sklearn pipeline
    pipeline, X_train, y_train = fit_pipeline(df_clean, target_col=target_col)

    # Persist for inference
    if save_path is not None:
        save_pipeline(pipeline, path=save_path)

    return pipeline, X_train, y_train


# ============================================================================
# Standalone smoke-test
# Run: python src/preprocessing.py
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # Build a tiny synthetic DataFrame that mimics the Telco schema
    sample = pd.DataFrame({
        "customerID":      ["001", "002", "003"],
        "gender":          ["Male", "Female", "Male"],
        "SeniorCitizen":   [0, 1, 0],
        "Partner":         ["Yes", "No", "Yes"],
        "Dependents":      ["No", "No", "Yes"],
        "tenure":          [12, 24, 6],
        "PhoneService":    ["Yes", "Yes", "No"],
        "MultipleLines":   ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity":  ["Yes", "No", "No internet service"],
        "OnlineBackup":    ["No", "Yes", "No internet service"],
        "DeviceProtection":["No", "No", "No internet service"],
        "TechSupport":     ["No", "No", "No internet service"],
        "StreamingTV":     ["No", "No", "No internet service"],
        "StreamingMovies": ["No", "No", "No internet service"],
        "Contract":        ["Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling":["Yes", "No", "Yes"],
        "PaymentMethod":   ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges":  [55.0, 80.0, 25.0],
        "TotalCharges":    ["660", "1920", " "],   # whitespace = hidden null
        "Churn":           ["No", "Yes", "No"],
    })

    try:
        pipeline, X, y = run_preprocessing(sample, save_path=None)
        print(f"\n✓  run_preprocessing() succeeded.")
        print(f"   X shape : {X.shape}")
        print(f"   y values: {y}")
        print(f"\nPipeline steps:")
        for name, transformer, cols in pipeline.transformers_:
            print(f"  [{name}] → {type(transformer).__name__}  columns: {cols}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
