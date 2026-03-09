"""
utils.py
--------
Shared helper utilities used across the pipeline.

Includes:
  - Logging configuration (call once at app startup).
  - Config loader (reads from environment variables or a JSON config file).
  - Timer context manager for benchmarking pipeline steps.
  - Data saving helper.

ML / Data Eng / MLOps role:
  Cross-cutting concerns that every module needs but shouldn't 
  re-implement themselves.  Centralising here reduces code duplication.
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure root logger.  Call this once in main scripts / API startup.

    Parameters
    ----------
    level    : Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    log_file : Optional path to write logs to a file in addition to stdout.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,   # re-applies config even if root logger was already set
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration.

    Priority (highest → lowest):
      1. Environment variables (prefixed with CHURN_)
      2. JSON config file at config_path
      3. Hard-coded defaults

    Returns
    -------
    dict with keys: data_path, model_name, test_size, random_state,
                    mlflow_uri, log_level
    """
    defaults: Dict[str, Any] = {
        "data_path":    "data/raw/churn.csv",
        "processed_path": "data/processed/churn_processed.csv",
        "model_name":   "best_model",
        "test_size":    0.2,
        "random_state": 42,
        "mlflow_uri":   "mlruns",
        "log_level":    "INFO",
        "api_host":     "0.0.0.0",
        "api_port":     8000,
    }

    # Overlay JSON config if provided
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            file_cfg = json.load(f)
        defaults.update(file_cfg)

    # Overlay environment variables (CHURN_DATA_PATH → data_path, etc.)
    env_map = {
        "CHURN_DATA_PATH":    "data_path",
        "CHURN_MODEL_NAME":   "model_name",
        "CHURN_TEST_SIZE":    "test_size",
        "CHURN_RANDOM_STATE": "random_state",
        "MLFLOW_TRACKING_URI": "mlflow_uri",
        "LOG_LEVEL":          "log_level",
    }
    for env_key, cfg_key in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            defaults[cfg_key] = val

    return defaults


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str = "Block") -> Generator[None, None, None]:
    """
    Simple wall-clock timer.

    Usage:
        with timer("Training"):
            model.fit(X, y)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.getLogger(__name__).info("%s completed in %.2f seconds.", label, elapsed)


# ---------------------------------------------------------------------------
# Data persistence helper
# ---------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV, creating parent directories if needed.

    Parameters
    ----------
    df    : DataFrame to save.
    path  : Destination file path.
    index : Whether to write row indices.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logging.getLogger(__name__).info("DataFrame saved → %s  (%d rows)", path, len(df))
