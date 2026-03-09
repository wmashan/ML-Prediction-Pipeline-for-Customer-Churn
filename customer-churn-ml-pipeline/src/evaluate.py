"""
evaluate.py
-----------
Produces a full evaluation report for any trained sklearn classifier.

Metrics included:
  - Accuracy, Precision, Recall, F1-score (weighted)
  - ROC-AUC
  - Confusion Matrix
  - Classification Report (per-class breakdown)

ML role       : Quantifies model quality beyond raw accuracy.
                Churn is class-imbalanced → ROC-AUC and recall matter more.
Data Eng role : Evaluation results can be written back to a reporting table.
MLOps role    : Metric thresholds act as quality gates in CI pipelines.
                If ROC-AUC < threshold → deployment is blocked.
"""

import logging
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")   # non-interactive backend – safe in Docker/CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Evaluate a fitted classifier and return a metrics dictionary.

    Parameters
    ----------
    model      : Fitted sklearn estimator with predict / predict_proba.
    X_test     : Transformed feature array (numpy).
    y_test     : True labels (numpy array of 0/1).
    model_name : Label shown in log messages.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }

    logger.info("=== %s Evaluation ===", model_name)
    for k, v in metrics.items():
        logger.info("  %-12s : %.4f", k, v)

    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "models/confusion_matrix.png",
) -> None:
    """
    Save a labelled confusion matrix heatmap as a PNG file.

    Parameters
    ----------
    save_path : Destination for the PNG file.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)


def plot_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "models/roc_curve.png",
) -> None:
    """
    Save a ROC curve plot as a PNG file.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved → %s", save_path)


# ---------------------------------------------------------------------------
# Quality gate – used by CI to block bad deployments
# ---------------------------------------------------------------------------

def assert_minimum_quality(
    metrics: Dict[str, float],
    min_roc_auc: float = 0.75,
    min_recall: float = 0.60,
) -> None:
    """
    Raise an AssertionError if the model does not meet minimum thresholds.

    Parameters
    ----------
    min_roc_auc : Minimum acceptable ROC-AUC score.
    min_recall  : Minimum acceptable recall for churn class.

    Raises
    ------
    AssertionError if either gate is not met.
    """
    assert metrics["roc_auc"] >= min_roc_auc, (
        f"ROC-AUC {metrics['roc_auc']:.4f} < threshold {min_roc_auc}"
    )
    assert metrics["recall"] >= min_recall, (
        f"Recall {metrics['recall']:.4f} < threshold {min_recall}"
    )
    logger.info("Quality gate PASSED (ROC-AUC=%.4f, Recall=%.4f)", metrics["roc_auc"], metrics["recall"])
