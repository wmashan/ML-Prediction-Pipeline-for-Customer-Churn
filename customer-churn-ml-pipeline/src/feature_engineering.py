"""
feature_engineering.py
-----------------------
Transforms cleaned data into model-ready features.

Responsibilities:
  1. Encode categorical variables (Label / One-Hot).
  2. Scale numeric features (StandardScaler / MinMaxScaler).
  3. Create new derived features that may improve model accuracy.
  4. Split data into train / test sets.

ML role       : Better features → better model performance.
Data Eng role : Builds the feature store input; transforms are repeatable.
MLOps role    : Sklearn Pipeline wraps transforms so inference reuses
                the SAME fitted scaler/encoder — no training/serving skew.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature creation helpers
# ---------------------------------------------------------------------------

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new columns derived from existing ones.

    Examples for a Telco churn dataset:
      - tenure_group   : bucket tenure into short / mid / long-term customers
      - avg_monthly_spend : totalcharges / (tenure + 1)
    """
    df = df.copy()

    if "tenure" in df.columns:
        bins = [0, 12, 24, 48, 72, np.inf]
        labels = ["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6yr+"]
        df["tenure_group"] = pd.cut(
            df["tenure"], bins=bins, labels=labels, right=False
        ).astype(str)
        logger.debug("Created 'tenure_group' feature.")

    if "totalcharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_spend"] = df["totalcharges"] / (df["tenure"] + 1)
        logger.debug("Created 'avg_monthly_spend' feature.")

    return df


# ---------------------------------------------------------------------------
# Column categorisation helper
# ---------------------------------------------------------------------------

def get_column_groups(
    df: pd.DataFrame,
    target_col: str = "churn",
    id_cols: list[str] | None = None,
) -> Tuple[list[str], list[str]]:
    """
    Return (numeric_features, categorical_features) excluding target and IDs.
    """
    if id_cols is None:
        id_cols = ["customerid"]   # common in Telco datasets

    drop_cols = [target_col] + [c for c in id_cols if c in df.columns]
    feature_df = df.drop(columns=drop_cols, errors="ignore")

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info("Numeric features  : %s", numeric_cols)
    logger.info("Categorical features: %s", cat_cols)
    return numeric_cols, cat_cols


# ---------------------------------------------------------------------------
# Sklearn ColumnTransformer builder
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_cols: list[str],
    cat_cols: list[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Scales numeric columns with StandardScaler
      - One-hot encodes categorical columns (drops first to avoid multicollinearity)

    Returns a FITTED-ready transformer (call .fit_transform on training data).
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    return preprocessor


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    target_col: str = "churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from preprocessing step.
    target_col : str
        Column to predict.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Ensures reproducible splits.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    preprocessor : fitted ColumnTransformer (save this for inference!)
    """
    logger.info("Starting feature engineering.")

    df = create_derived_features(df)

    # Separate features and target
    X = df.drop(columns=[target_col, "customerid"], errors="ignore")
    y = df[target_col].values

    # Split BEFORE fitting transforms → prevents data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    numeric_cols, cat_cols = get_column_groups(df, target_col=target_col)

    # Remove extra columns that might appear after split
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    preprocessor = build_preprocessor(numeric_cols, cat_cols)

    X_train = preprocessor.fit_transform(X_train)   # fit on train only
    X_test = preprocessor.transform(X_test)          # apply same transform to test

    logger.info(
        "Feature engineering done. X_train: %s | X_test: %s",
        X_train.shape, X_test.shape
    )

    return X_train, X_test, y_train, y_test, preprocessor
