"""
data_ingestion.py
-----------------
Entry point of the ML pipeline. Loads raw CSV data from disk into a
Pandas DataFrame, validates the file, checks the schema, and returns
a clean DataFrame ready for preprocessing.

Responsibilities (Single Responsibility Principle):
  - Load data only. Does NOT clean, transform, or encode.
  - Validate that the file and expected columns exist.
  - Log metadata so every pipeline run is traceable.

─────────────────────────────────────────────
How this fits into a Data Engineering pipeline
─────────────────────────────────────────────
In a production data pipeline, data flows through distinct zones:

   [Source]  →  [Ingestion]  →  [Raw Zone]  →  [Processed Zone]  →  [Feature Store]  →  [Model]

This file is the INGESTION layer:
  - It is the only file allowed to talk to the data source (CSV, S3, DB).
  - All downstream modules (preprocessing, feature engineering) receive a
    DataFrame from here — they never touch the file system directly.
  - This isolation means you can swap "load from CSV" for "load from PostgreSQL"
    in one place without touching any other file.

MLOps role:
  - The data path is configurable (env var / config file) so CI/CD can
    override it to point at a test dataset without changing code.
  - Schema validation here acts as an early quality gate — bad data fails
    loudly at ingestion rather than silently corrupting the trained model.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger.
# Every module gets its OWN named logger (not the root logger).
# This means log lines show exactly which file produced them:
#   e.g.  "src.data_ingestion | INFO | Loaded 7043 rows × 21 columns"
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected schema — columns that MUST exist for the pipeline to work.
# Edit this list if your CSV has different column names.
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: List[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",          # target column — must be present
]


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Read a CSV file from disk into a Pandas DataFrame.

    This function does one thing only: read the file.
    It does NOT clean, transform, or encode any values.

    Parameters
    ----------
    file_path : str | Path
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw data exactly as it appears in the CSV.

    Raises
    ------
    FileNotFoundError
        If no file exists at the given path.
    ValueError
        If the file loads as an empty DataFrame.

    Example
    -------
    >>> df = load_csv("data/raw/churn.csv")
    >>> print(df.shape)    # (7043, 21)
    """
    path = Path(file_path)

    # ── Validate file exists ───────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path.resolve()}\n"
            f"Download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            f"and place the file at: data/raw/churn.csv"
        )

    # ── Load ──────────────────────────────────────────────────────────────
    logger.info("Loading data from: %s", path.resolve())
    df = pd.read_csv(path)

    # ── Validate not empty ────────────────────────────────────────────────
    if df.empty:
        raise ValueError(f"File loaded successfully but contains zero rows: {path.resolve()}")

    logger.info("Loaded %d rows × %d columns", df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Schema validator
# ---------------------------------------------------------------------------

def validate_schema(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> None:
    """
    Check that all required columns are present in the DataFrame.

    Why this matters in production:
      If upstream (data team, Kaggle download, DB export) drops or renames
      a column, the pipeline fails here with a clear error message —
      rather than failing silently 3 steps later with a cryptic KeyError.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded raw DataFrame.
    required_columns : list[str] | None
        Columns that must exist. Defaults to REQUIRED_COLUMNS.

    Raises
    ------
    ValueError
        With a list of all missing columns so you can fix them in one go.
    """
    expected = required_columns if required_columns is not None else REQUIRED_COLUMNS
    actual_cols = set(df.columns.tolist())
    missing = [col for col in expected if col not in actual_cols]

    if missing:
        raise ValueError(
            f"Schema validation failed — missing {len(missing)} required column(s):\n"
            f"  {missing}\n"
            f"Columns found in CSV:\n  {sorted(actual_cols)}"
        )

    logger.info("Schema validation passed — all %d required columns present.", len(expected))


# ---------------------------------------------------------------------------
# Metadata logger (useful for MLflow / audit trail)
# ---------------------------------------------------------------------------

def log_data_summary(df: pd.DataFrame) -> Dict[str, object]:
    """
    Log and return a metadata dictionary about the loaded dataset.

    The returned dict can be passed directly to mlflow.log_params()
    to record data characteristics alongside every experiment run.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys: n_rows, n_cols, n_missing, churn_rate_pct, source_columns
    """
    n_missing = int(df.isnull().sum().sum())

    # Churn rate — handles both "Yes"/"No" string and 0/1 int encodings
    # Use is_numeric_dtype instead of dtype == object so this works on both
    # the legacy object dtype (pandas < 3) and StringDtype (pandas ≥ 3).
    churn_rate: Optional[float] = None
    if "Churn" in df.columns:
        col = df["Churn"]
        if pd.api.types.is_numeric_dtype(col):
            churn_rate = round(float(col.mean()) * 100, 2)
        else:
            churn_rate = round((col.astype(str).str.lower() == "yes").mean() * 100, 2)

    summary: Dict[str, object] = {
        "n_rows":          df.shape[0],
        "n_cols":          df.shape[1],
        "n_missing":       n_missing,
        "churn_rate_pct":  churn_rate,
        "source_columns":  df.columns.tolist(),
    }

    logger.info("── Dataset Summary ──────────────────────")
    logger.info("  Rows            : %d", summary["n_rows"])
    logger.info("  Columns         : %d", summary["n_cols"])
    logger.info("  Total nulls     : %d", summary["n_missing"])
    if churn_rate is not None:
        logger.info("  Churn rate      : %.2f%%", churn_rate)
    logger.info("────────────────────────────────────────")

    return summary


# ---------------------------------------------------------------------------
# Main orchestrator — this is what the rest of the pipeline imports
# ---------------------------------------------------------------------------

def ingest(
    file_path: str | Path = "data/raw/churn.csv",
    validate: bool = True,
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Full ingestion pipeline: load → validate schema → log summary.

    This is the single function all other modules call.
    Using one entry point means you only change this file (not 5 others)
    if the data source ever changes (e.g. CSV → PostgreSQL → S3).

    Parameters
    ----------
    file_path : str | Path
        Path to the raw CSV file.
    validate : bool
        If True, runs schema validation. Set False only in tests with mock data.
    required_columns : list[str] | None
        Override the default REQUIRED_COLUMNS list if needed.

    Returns
    -------
    pd.DataFrame
        Raw data, schema-validated, ready to pass to preprocessing.py.

    Usage
    -----
    # In train.py, preprocessing.py, or any pipeline script:
    from src.data_ingestion import ingest

    df = ingest("data/raw/churn.csv")
    # df is now a clean, validated raw DataFrame
    """
    df = load_csv(file_path)

    if validate:
        validate_schema(df, required_columns)

    log_data_summary(df)

    return df


# ---------------------------------------------------------------------------
# Standalone smoke-test
# Run directly to verify the pipeline entry point works:
#   python src/data_ingestion.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")      # ensure src/ imports resolve from project root

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    try:
        data = ingest("data/raw/churn.csv")
        print("\nSample rows:")
        print(data.head(3).to_string())
        print(f"\n✓  data_ingestion.py is working correctly. Shape: {data.shape}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nDownload the Telco Churn dataset from Kaggle and place it at data/raw/churn.csv")
