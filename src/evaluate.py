"""
evaluate.py
-----------
Produces a full, reusable evaluation report for any trained sklearn classifier.

What this module does
---------------------
  1. Computes all standard binary classification metrics.
  2. Prints a human-readable confusion matrix to stdout.
  3. Prints a per-class classification report to stdout.
  4. Optionally saves metrics to a JSON file (useful for CI dashboards).
  5. Saves visualisations: confusion matrix heatmap, ROC curve,
     and precision-recall curve.
  6. Runs a quality gate — raises an error if the model is below threshold
     (used in CI/CD pipelines to block a bad model from being deployed).

ML role
-------
  Numbers alone don't tell the full story. This module surfaces *which*
  churners the model catches, which it misses, and at what cost to
  precision — all critical before you hand results to a business team.

Data Engineering role
---------------------
  Metrics saved to JSON can be read back by a reporting pipeline or
  logged into a database as part of a model-tracking workflow.

MLOps role
----------
  The quality gate function is called in CI (GitHub Actions) after
  training.  If ROC-AUC < 0.75 or Recall < 0.60, the pipeline fails
  and the bad model is never written to the model registry.

Which metrics matter most for churn prediction — see bottom of file.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")    # non-interactive backend — safe in Docker / CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)

# Default directory for saving artefacts
REPORTS_DIR = Path("models")


# ============================================================================
# 1. Core metrics computation
# ============================================================================

def compute_metrics(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all standard binary classification metrics and return them as a
    plain dictionary.

    This function is intentionally pure: it has no side effects
    (no printing, no file I/O). Call the printing / saving helpers
    below if you need those.

    Parameters
    ----------
    model  : Fitted sklearn estimator with .predict() and .predict_proba().
    X_test : Transformed 2-D numpy array (output of preprocessing pipeline).
    y_test : 1-D numpy array of true binary labels (0 = no churn, 1 = churn).

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, roc_auc, avg_precision

    Metric definitions
    ------------------
    accuracy       : (TP + TN) / total.  Percentage of all correct predictions.
    precision      : TP / (TP + FP).     Of customers flagged as churners,
                                         how many actually churned?
    recall         : TP / (TP + FN).     Of customers who actually churned,
                                         how many did we catch?
    f1             : 2 * P * R / (P + R). Harmonic mean — rewards models that
                                         are both precise AND recall-high.
    roc_auc        : Area under the ROC curve (threshold-free ranking metric).
    avg_precision  : Area under the precision-recall curve.  Better than
                     ROC-AUC when the positive class is rare (churn ~26%).
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics: Dict[str, float] = {
        "accuracy":      round(float(accuracy_score(y_test, y_pred)),             4),
        "precision":     round(float(precision_score(y_test, y_pred,
                                                     zero_division=0)),           4),
        "recall":        round(float(recall_score(y_test, y_pred,
                                                  zero_division=0)),              4),
        "f1":            round(float(f1_score(y_test, y_pred,
                                              zero_division=0)),                  4),
        "roc_auc":       round(float(roc_auc_score(y_test, y_prob)),             4),
        "avg_precision": round(float(average_precision_score(y_test, y_prob)),   4),
    }
    return metrics


# ============================================================================
# 2. Console printing helpers
# ============================================================================

def print_metrics(
    metrics: Dict[str, float],
    model_name: str = "Model",
) -> None:
    """
    Print all metrics in an aligned table to stdout.

    Example output
    --------------
    ┌─────────────────────────────────────────┐
    │  Model Evaluation — random_forest       │
    ├──────────────────────┬──────────────────┤
    │  Accuracy            │  0.8012          │
    │  Precision           │  0.5325          │
    │  Recall              │  0.7241          │
    │  F1 Score            │  0.6143          │
    │  ROC-AUC             │  0.8712          │
    │  Avg Precision (PR)  │  0.6501          │
    └──────────────────────┴──────────────────┘
    """
    labels = {
        "accuracy":      "Accuracy",
        "precision":     "Precision",
        "recall":        "Recall",
        "f1":            "F1 Score",
        "roc_auc":       "ROC-AUC        ★",  # ★ = primary metric
        "avg_precision": "Avg Precision (PR)",
    }

    name_w  = max(len(v) for v in labels.values()) + 2
    value_w = 10
    total_w = name_w + value_w + 3   # 3 for " │ "

    border_top = "┌" + "─" * total_w + "┐"
    border_mid = "├" + "─" * (name_w + 2) + "┬" + "─" * (value_w + 2) + "┤"
    border_bot = "└" + "─" * (name_w + 2) + "┴" + "─" * (value_w + 2) + "┘"
    title      = f"│  Model Evaluation — {model_name}"

    print("\n" + border_top)
    print(title.ljust(total_w + 1) + "│")
    print(border_mid)
    for key, label in labels.items():
        val_str = f"{metrics[key]:.4f}"
        row = f"│  {label:<{name_w}}│  {val_str:<{value_w}}│"
        print(row)
    print(border_bot)


def print_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: tuple = ("No Churn", "Churn"),
) -> None:
    """
    Print a text confusion matrix to stdout.

    A confusion matrix shows how predictions break down across the four
    possible outcomes:

                    Predicted No    Predicted Churn
    Actual No   │   TN (correct)  │  FP (false alarm)  │
    Actual Churn│   FN (missed!)  │  TP (correct)      │

    For a churn retention campaign:
      - FP (false alarm) = we offer a discount to a loyal customer.  Wasteful.
      - FN (missed)      = a churner leaves without any intervention.  Costly.

    Parameters
    ----------
    labels : Display names for class 0 and class 1.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    col_w = max(len(labels[0]), len(labels[1]), 8) + 2

    header = (
        f"\n{'Confusion Matrix':^{col_w * 3}}\n"
        + " " * col_w
        + f"{'Predicted':^{col_w * 2}}\n"
        + " " * col_w
        + f"{labels[0]:^{col_w}}{labels[1]:^{col_w}}"
    )
    row0 = f"{'Actual ' + labels[0]:<{col_w}}{tn:^{col_w}}{fp:^{col_w}}"
    row1 = f"{'Actual ' + labels[1]:<{col_w}}{fn:^{col_w}}{tp:^{col_w}}"
    divider = "─" * (col_w * 3)

    print(header)
    print(divider)
    print(row0)
    print(row1)
    print(divider)
    print(
        f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}  "
        f"(missed churners: {fn}, false alarms: {fp})\n"
    )


def print_classification_report(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Print sklearn's per-class classification report to stdout.

    This shows precision, recall, and F1 broken out separately for
    class 0 (No Churn) and class 1 (Churn), plus macro and weighted
    averages.  Useful for spotting if the model ignores one class entirely.
    """
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=["No Churn (0)", "Churn (1)"],
    )
    print("\nClassification Report")
    print("─" * 60)
    print(report)


# ============================================================================
# 3. JSON export
# ============================================================================

def save_metrics_json(
    metrics: Dict[str, float],
    path: Optional[Path] = None,
    model_name: str = "model",
) -> Path:
    """
    Save the metrics dictionary to a JSON file.

    Why save metrics to JSON?
      - CI pipelines can read the file and compare against previous runs.
      - A reporting dashboard can ingest it without re-running the model.
      - Provides a human-readable audit trail alongside the .joblib file.

    Parameters
    ----------
    metrics    : Dictionary returned by compute_metrics().
    path       : Exact file path.  Defaults to models/<model_name>_metrics.json.
    model_name : Used to build the default filename.

    Returns
    -------
    Path : The location where the file was saved.
    """
    if path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"{model_name}_metrics.json"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"model": model_name, "metrics": metrics}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Metrics saved → %s", path)
    return path


# ============================================================================
# 4. Visualisation helpers (each saves a PNG)
# ============================================================================

def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[Path] = None,
    model_name: str = "model",
) -> Path:
    """
    Save a colour-coded confusion matrix heatmap as a PNG.

    Uses seaborn for nicer formatting than sklearn's default.
    The colour intensity reflects the count in each cell, making
    it easy to spot if one cell (e.g. false-negatives) is unusually large.

    Returns the path where the PNG was saved.
    """
    if save_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = REPORTS_DIR / f"{model_name}_confusion_matrix.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

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
        linewidths=0.5,
        linecolor="grey",
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info("Confusion matrix plot saved → %s", save_path)
    return save_path


def plot_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[Path] = None,
    model_name: str = "model",
) -> Path:
    """
    Save a ROC (Receiver Operating Characteristic) curve as a PNG.

    The ROC curve plots True Positive Rate vs False Positive Rate at
    every possible classification threshold.  The AUC (area under the
    curve) summarises performance in a single number:
      0.5 = chance, 0.7 = acceptable, 0.85+ = good, 1.0 = perfect.

    Returns the path where the PNG was saved.
    """
    if save_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = REPORTS_DIR / f"{model_name}_roc_curve.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(
        model, X_test, y_test,
        name=model_name,
        ax=ax,
        color="steelblue",
    )
    # Diagonal reference line (random classifier baseline)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info("ROC curve saved → %s", save_path)
    return save_path


def plot_precision_recall_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[Path] = None,
    model_name: str = "model",
) -> Path:
    """
    Save a Precision-Recall curve as a PNG.

    Why plot a PR curve?
      For imbalanced datasets the PR curve is often more informative than
      the ROC curve.  The ROC curve can look optimistic because it accounts
      for the large pool of true negatives.  The PR curve focuses only on
      the positive (churn) class, showing the trade-off between catching
      more churners (higher recall) and flagging fewer false alarms
      (lower false-positive rate = higher precision).

    The baseline for a PR curve is the prevalence of the positive class
    (~ 0.26 for Telco churn).  Any model above this horizontal line is
    useful.

    Returns the path where the PNG was saved.
    """
    if save_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = REPORTS_DIR / f"{model_name}_pr_curve.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    prevalence = float(y_test.mean())   # baseline = random classifier

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_test, y_prob,
        name=model_name,
        ax=ax,
        color="darkorange",
    )
    ax.axhline(
        y=prevalence, color="grey", linestyle="--", lw=1,
        label=f"Random baseline (prevalence = {prevalence:.2f})",
    )
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=12, pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(visible=True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info("Precision-Recall curve saved → %s", save_path)
    return save_path


# ============================================================================
# 5. Quality gate — used by CI to block bad deployments
# ============================================================================

def assert_minimum_quality(
    metrics: Dict[str, float],
    min_roc_auc: float = 0.75,
    min_recall: float = 0.60,
) -> None:
    """
    Raise AssertionError if the model is below minimum quality thresholds.

    This is called as the final step in a CI pipeline.  If either check
    fails, the GitHub Actions job exits with a non-zero code, preventing
    the model from being pushed to the model registry or deployed.

    Thresholds (defaults)
    ---------------------
    min_roc_auc = 0.75 : Model must rank churners above non-churners
                          with reasonable consistency.
    min_recall  = 0.60 : Model must catch at least 60% of actual churners.
                          Missing 40% of churners means wasted opportunity
                          — too many customers leave without intervention.

    Parameters
    ----------
    metrics     : Dict returned by compute_metrics().
    min_roc_auc : Minimum acceptable ROC-AUC.
    min_recall  : Minimum acceptable recall on the churn class.

    Raises
    ------
    AssertionError with a descriptive message if either gate fails.
    """
    roc_auc = metrics["roc_auc"]
    recall  = metrics["recall"]

    if roc_auc < min_roc_auc:
        raise AssertionError(
            f"Quality gate FAILED: ROC-AUC {roc_auc:.4f} "
            f"< minimum {min_roc_auc:.2f}. "
            "Model is not good enough to deploy."
        )
    if recall < min_recall:
        raise AssertionError(
            f"Quality gate FAILED: Recall {recall:.4f} "
            f"< minimum {min_recall:.2f}. "
            "Model misses too many churners."
        )

    logger.info(
        "Quality gate PASSED — ROC-AUC=%.4f (≥%.2f), Recall=%.4f (≥%.2f).",
        roc_auc, min_roc_auc, recall, min_recall,
    )


# ============================================================================
# 6. Full evaluation report — orchestrates everything above
# ============================================================================

def full_evaluation_report(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    save_dir: Optional[Path] = None,
    save_json: bool = True,
    save_plots: bool = True,
    run_quality_gate: bool = False,
    min_roc_auc: float = 0.75,
    min_recall:  float = 0.60,
) -> Dict[str, float]:
    """
    Run the complete evaluation pipeline for one model.

    Steps
    -----
    1. Compute all metrics with compute_metrics().
    2. Print the metrics table to stdout.
    3. Print the text confusion matrix to stdout.
    4. Print the per-class classification report to stdout.
    5. If save_json=True  → write metrics to JSON.
    6. If save_plots=True → save confusion matrix, ROC, and PR curve PNGs.
    7. If run_quality_gate=True → raise on failure.

    Parameters
    ----------
    model         : Fitted sklearn estimator.
    X_test        : Transformed test features (numpy array).
    y_test        : True binary labels (numpy array).
    model_name    : Used in titles, filenames, and the JSON payload.
    save_dir      : Directory for JSON and PNGs. Defaults to models/.
    save_json     : Whether to write a metrics JSON file.
    save_plots    : Whether to generate and save PNG visualisations.
    run_quality_gate : Whether to raise if metrics are below thresholds.
    min_roc_auc   : Threshold used if run_quality_gate=True.
    min_recall    : Threshold used if run_quality_gate=True.

    Returns
    -------
    Dict[str, float] — the metrics dictionary.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = REPORTS_DIR

    # ── 1. Compute ────────────────────────────────────────────────────────
    metrics = compute_metrics(model, X_test, y_test)

    # ── 2. Print metrics table ────────────────────────────────────────────
    print_metrics(metrics, model_name=model_name)

    # ── 3. Print confusion matrix (text) ──────────────────────────────────
    print_confusion_matrix(model, X_test, y_test)

    # ── 4. Print classification report ───────────────────────────────────
    print_classification_report(model, X_test, y_test)

    # ── 5. Save JSON ──────────────────────────────────────────────────────
    if save_json:
        save_metrics_json(
            metrics,
            path=save_dir / f"{model_name}_metrics.json",
            model_name=model_name,
        )

    # ── 6. Save plots ─────────────────────────────────────────────────────
    if save_plots:
        plot_confusion_matrix(
            model, X_test, y_test,
            save_path=save_dir / f"{model_name}_confusion_matrix.png",
            model_name=model_name,
        )
        plot_roc_curve(
            model, X_test, y_test,
            save_path=save_dir / f"{model_name}_roc_curve.png",
            model_name=model_name,
        )
        plot_precision_recall_curve(
            model, X_test, y_test,
            save_path=save_dir / f"{model_name}_pr_curve.png",
            model_name=model_name,
        )

    # ── 7. Quality gate ───────────────────────────────────────────────────
    if run_quality_gate:
        assert_minimum_quality(
            metrics,
            min_roc_auc=min_roc_auc,
            min_recall=min_recall,
        )

    return metrics


# ============================================================================
# Standalone smoke-test   (python src/evaluate.py)
# ============================================================================

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # ── Build a tiny synthetic dataset ───────────────────────────────────
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=6,
        weights=[0.74, 0.26],   # mimic 26% churn prevalence
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(class_weight="balanced", max_iter=500)),
    ])
    model.fit(X_train, y_train)

    # ── Run full report into a temp directory ────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = full_evaluation_report(
            model, X_test, y_test,
            model_name="logistic_regression_smoke",
            save_dir=Path(tmpdir),
            save_json=True,
            save_plots=True,
            run_quality_gate=False,   # synthetic data won't hit thresholds
        )

    print("\n✓ evaluate.py smoke-test PASSED")
    print(f"  Final metrics: {metrics}")
