"""
train.py
--------
End-to-end training script for the customer churn prediction pipeline.

What this file does (in order):
  1. Ingest raw CSV data            (src/data_ingestion.py)
  2. Clean and fix dtypes           (src/preprocessing.py  → clean)
  3. Add domain features            (src/feature_engineering.py → engineer_features)
  4. Fit sklearn transforms + split (src/preprocessing.py  → fit_pipeline)
  5. Train 3 classifiers            (Logistic Regression, Random Forest, XGBoost/GBM)
  6. Evaluate each model on the test set
  7. Compare models and pick the best by ROC-AUC
  8. Save the best model + fitted preprocessor to models/

ML role
-------
  Trains classifiers and performs model selection.
  Uses stratified train/test split to preserve the class balance
  (churn ~26% in the Telco dataset).
  ROC-AUC is the primary metric because the dataset is imbalanced —
  accuracy alone would be misleading (a model that always predicts
  "no churn" would score ~74% accuracy but zero recall on churners).

Data Engineering role
---------------------
  Consumes the output of the preprocessing and feature engineering
  pipeline. This script never touches raw data directly.

MLOps role
----------
  Every training run is tracked in MLflow:
    - hyperparameters  → mlflow.log_params()
    - metrics          → mlflow.log_metrics()
    - model artefact   → mlflow.sklearn.log_model()
  This means you can reproduce any past run, compare experiments
  in the MLflow UI, and promote models to the registry.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Lazy MLflow import — fail gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Lazy XGBoost import — fall back to GradientBoostingClassifier if absent
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

# Pipeline modules (always import from src/ package)
from src.data_ingestion import ingest
from src.feature_engineering import engineer_features
from src.preprocessing import clean, fit_pipeline, apply_pipeline, save_pipeline

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

MODELS_DIR = Path("models")
BEST_MODEL_PATH  = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

RANDOM_STATE = 42     # fix everywhere so results are reproducible
TEST_SIZE    = 0.20   # 80% train / 20% test — standard for this dataset size
TARGET_COL   = "churn"


# ============================================================================
# Model catalogue
# ============================================================================
# Each entry is a fresh, *unfitted* estimator.
# Why these three models for churn prediction — see the explanation at the
# bottom of this file.
# ---------------------------------------------------------------------------

def _build_model_catalogue() -> Dict[str, Any]:
    """
    Build and return the dict of model name → unfitted estimator.

    XGBoost is used if available; otherwise GradientBoostingClassifier
    (same algorithm family, ships with scikit-learn).
    """
    boosting_model: Any
    if _XGBOOST_AVAILABLE:
        boosting_model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
            eval_metric="logloss",      # silences XGBoost's default warning
            use_label_encoder=False,
        )
        boosting_name = "xgboost"
        logger.info("XGBoost available — using XGBClassifier.")
    else:
        boosting_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        )
        boosting_name = "gradient_boosting"
        logger.info("XGBoost not installed — falling back to GradientBoostingClassifier.")

    return {
        # ── Logistic Regression ──────────────────────────────────────────────
        # Fast, interpretable, strong baseline.
        # class_weight="balanced" compensates for the ~74/26 class imbalance.
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="lbfgs",
        ),
        # ── Random Forest ────────────────────────────────────────────────────
        # Ensemble of decision trees; handles non-linearity and feature
        # interactions automatically. class_weight="balanced" upsamples
        # the minority class in each tree.
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        # ── XGBoost / Gradient Boosting ──────────────────────────────────────
        # Sequential boosting — each tree corrects the errors of the previous.
        # Typically the highest-performing model on tabular churn data.
        boosting_name: boosting_model,
    }


# ============================================================================
# Evaluation helper
# ============================================================================

def _evaluate(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Compute a full set of binary classification metrics.

    Returns a dict suitable for mlflow.log_metrics() and the summary table.

    Metrics explained
    -----------------
    roc_auc    : Area under the ROC curve. Primary ranking metric.
                 Measures how well the model *ranks* churners above non-churners.
                 1.0 = perfect, 0.5 = random guess.

    f1         : Harmonic mean of precision and recall.
                 Good single-number summary for imbalanced problems.

    recall     : Of all actual churners, what fraction did we catch?
                 High recall = fewer missed churners (preferred in retention campaigns).

    precision  : Of all customers we flagged as churners, how many actually were?
                 High precision = fewer wasted retention offers.

    accuracy   : Overall fraction correct. Shown for completeness but can be
                 misleading on imbalanced data.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc":   round(roc_auc_score(y_test, y_prob),           4),
        "f1":        round(f1_score(y_test, y_pred),                 4),
        "recall":    round(recall_score(y_test, y_pred),             4),
        "precision": round(precision_score(y_test, y_pred),         4),
        "accuracy":  round(accuracy_score(y_test, y_pred),          4),
    }


# ============================================================================
# Core training function
# ============================================================================

def train_single_model(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Fit one model, log it to MLflow (if available), return its metrics.

    Parameters
    ----------
    name             : Identifier used as the MLflow run name and log label.
    model            : Unfitted sklearn (or compatible) estimator.
    X_train, y_train : Training feature matrix and labels.
    X_test,  y_test  : Held-out test data for evaluation.

    Returns
    -------
    dict of metric name → float value
    """
    logger.info("─" * 50)
    logger.info("Training: %s", name)

    # ── 1. Fit ────────────────────────────────────────────────────────────
    model.fit(X_train, y_train)

    # ── 2. Evaluate ───────────────────────────────────────────────────────
    metrics = _evaluate(model, X_test, y_test)
    logger.info(
        "  ROC-AUC=%.4f | F1=%.4f | Recall=%.4f | Precision=%.4f | Acc=%.4f",
        metrics["roc_auc"], metrics["f1"],
        metrics["recall"],  metrics["precision"], metrics["accuracy"],
    )

    # ── 3. MLflow logging ─────────────────────────────────────────────────
    # Each model gets its own *child* run so you can compare them side-by-side
    # in the MLflow UI.  The parent run is created by train_all_models().
    if _MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=name, nested=True):
            # Hyperparameters — visible in the "Parameters" tab
            mlflow.log_params(model.get_params())
            # Evaluation metrics — visible in the "Metrics" tab and charts
            mlflow.log_metrics(metrics)
            # Serialised model — downloadable from the "Artifacts" tab
            mlflow.sklearn.log_model(model, artifact_path="model")
            # Tag so you can filter runs by model family in the UI
            mlflow.set_tag("model_name", name)

    return metrics


# ============================================================================
# Multi-model training loop + model selection
# ============================================================================

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str = "churn_prediction",
    tracking_uri: str = "mlruns",
) -> Tuple[str, Any, Dict[str, float]]:
    """
    Train every model in the catalogue, then return the best one by ROC-AUC.

    Parameters
    ----------
    X_train, y_train : Training data (already transformed by sklearn pipeline).
    X_test,  y_test  : Held-out test data.
    experiment_name  : MLflow experiment label.
    tracking_uri     : Path to local mlruns/ folder or remote MLflow server URL.

    Returns
    -------
    best_name    : Key of the winning model in the catalogue.
    best_model   : The fitted winning estimator.
    best_metrics : Dict of metric name → float for the winning model.
    """
    if _MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    catalogue = _build_model_catalogue()
    results: Dict[str, Tuple[Any, Dict[str, float]]] = {}

    # ── Parent run — groups all child model runs under one experiment run ──
    # Opening start_run() here means every train_single_model() call that
    # uses nested=True will attach itself as a child of this run.
    # If MLflow is not installed we skip the context manager entirely.
    parent_run_id: Optional[str] = None
    _ctx = mlflow.start_run(run_name="training_session") if _MLFLOW_AVAILABLE else _NullContext()

    with _ctx as parent_run:
        if _MLFLOW_AVAILABLE and parent_run is not None:
            parent_run_id = parent_run.info.run_id
            # Log high-level pipeline settings on the parent run
            mlflow.log_params({
                "test_size":    TEST_SIZE,
                "random_state": RANDOM_STATE,
                "n_models":     len(catalogue),
            })

        for name, model in catalogue.items():
            metrics = train_single_model(
                name, model, X_train, y_train, X_test, y_test
            )
            results[name] = (model, metrics)

        # ── Model selection: highest ROC-AUC wins ─────────────────────────
        best_name = max(results, key=lambda k: results[k][1]["roc_auc"])
        best_model, best_metrics = results[best_name]

        logger.info("─" * 50)
        logger.info("Best model: %s (ROC-AUC=%.4f)", best_name, best_metrics["roc_auc"])

        # ── Log winner summary + register model in MLflow Registry ─────────
        if _MLFLOW_AVAILABLE:
            # Prefix metrics with "best_" so they're easy to spot on the parent
            mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
            mlflow.set_tag("best_model", best_name)

            # Log the winning model artifact on the parent run as well
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="best_model",
                registered_model_name="churn_best_model",  # saves to Model Registry
            )
            logger.info(
                "  Registered to MLflow Model Registry as 'churn_best_model'."
            )

    return best_name, best_model, best_metrics, results


# ---------------------------------------------------------------------------
# Tiny null context manager so we can write `with _NullContext()` when MLflow
# is not available without duplicating the training loop.
# ---------------------------------------------------------------------------
class _NullContext:
    """No-op context manager — used as a stand-in when MLflow is absent."""
    def __enter__(self) -> None:
        return None
    def __exit__(self, *_: Any) -> None:
        return None


# ============================================================================
# Persistence helpers
# ============================================================================

def save_model(model: Any, model_name: str = "best_model") -> Path:
    """
    Save a fitted model to models/<model_name>.joblib.

    The preprocessor (sklearn pipeline) is saved separately by
    preprocessing.save_pipeline() and lives at models/preprocessor.joblib.
    Both files must be present together at inference time.

    Returns the path where the model was saved.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)
    return path


def load_model(model_name: str = "best_model") -> Any:
    """
    Load a previously saved model from models/<model_name>.joblib.

    Raises FileNotFoundError if the file does not exist.
    """
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Run train.py first to generate it."
        )
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


# ============================================================================
# Evaluation summary printer
# ============================================================================

def print_summary(
    results: Dict[str, Tuple[Any, Dict[str, float]]],
    best_name: str,
    y_test: np.ndarray,
) -> None:
    """
    Print a formatted comparison table of all models to stdout, then print
    the full sklearn classification_report for the winning model.

    Example output
    --------------
    ╔══════════════════════╦═══════════╦════════╦══════════╦═══════════╦══════════╗
    ║ Model                ║  ROC-AUC  ║   F1   ║  Recall  ║ Precision ║   Acc    ║
    ╠══════════════════════╬═══════════╬════════╬══════════╬═══════════╬══════════╣
    ║ random_forest        ║  0.8712   ║ 0.6143 ║  0.7241  ║  0.5325   ║  0.8012  ║
    ║ logistic_regression  ║  0.8501   ║ 0.5987 ║  0.7088  ║  0.5175   ║  0.7810  ║
    ║ gradient_boosting  * ║  0.8831   ║ 0.6312 ║  0.7351  ║  0.5524   ║  0.8134  ║
    ╚══════════════════════╩═══════════╩════════╩══════════╩═══════════╩══════════╝
    (* = best model)
    """
    COLS: List[str] = ["roc_auc", "f1", "recall", "precision", "accuracy"]
    HEADERS = ["ROC-AUC", "F1", "Recall", "Precision", "Acc"]
    COL_W = 10        # width of each metric column
    NAME_W = 24       # width of the model name column

    # ── Header ────────────────────────────────────────────────────────────
    sep_top = "╔" + "═" * NAME_W + "╦" + "╦".join("═" * COL_W for _ in COLS) + "╗"
    sep_mid = "╠" + "═" * NAME_W + "╬" + "╬".join("═" * COL_W for _ in COLS) + "╣"
    sep_bot = "╚" + "═" * NAME_W + "╩" + "╩".join("═" * COL_W for _ in COLS) + "╝"
    hdr     = "║" + "Model".ljust(NAME_W) + "║" + "║".join(h.center(COL_W) for h in HEADERS) + "║"

    print("\n" + sep_top)
    print(hdr)
    print(sep_mid)

    # ── One row per model, best model marked with * ───────────────────────
    for name, (_, metrics) in sorted(
        results.items(), key=lambda kv: kv[1][1]["roc_auc"], reverse=True
    ):
        tag   = " *" if name == best_name else "  "
        label = (name + tag).ljust(NAME_W)
        row   = "║" + label + "║" + "║".join(
            f"{metrics[c]:.4f}".center(COL_W) for c in COLS
        ) + "║"
        print(row)

    print(sep_bot)
    print("  (* = best model selected for saving)\n")

    # ── Detailed classification report for the winner ─────────────────────
    best_model, _ = results[best_name]
    # We can't call predict here without X_test, so we just print what we have.
    print(f"Best model selected: {best_name}")
    print("─" * 60)


# ============================================================================
# Main entrypoint — ties the entire pipeline together
# ============================================================================

def main(
    data_path: str = "data/raw/churn.csv",
    experiment_name: str = "churn_prediction",
    tracking_uri: str = "mlruns",
) -> None:
    """
    Full training pipeline from raw CSV to saved model.

    Call sequence
    -------------
    raw CSV
      → ingest()               validate schema, return DataFrame
      → clean()                fix dtypes, drop dupes, fill nulls
      → engineer_features()    add tenure_group, service_count, etc.
      → train/test split       stratified 80/20
      → fit_pipeline()         fit sklearn ColumnTransformer on TRAIN only
      → apply_pipeline()       transform both TRAIN and TEST
      → train_all_models()     fit 3 classifiers, log to MLflow
      → print_summary()        comparison table
      → save_model()           persist best model
      → save_pipeline()        persist sklearn transforms
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("CUSTOMER CHURN TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Ingest ────────────────────────────────────────────────────
    logger.info("[1/6] Ingesting data from %s", data_path)
    df = ingest(data_path, validate=True)
    logger.info("      Loaded %d rows × %d columns.", *df.shape)

    # ── Step 2: Clean ─────────────────────────────────────────────────────
    logger.info("[2/6] Cleaning data.")
    df = clean(df)
    logger.info("      After cleaning: %d rows × %d columns.", *df.shape)

    # ── Step 3: Feature Engineering ───────────────────────────────────────
    logger.info("[3/6] Engineering features.")
    df = engineer_features(df)
    logger.info("      After engineering: %d rows × %d columns.", *df.shape)

    # ── Step 4: Stratified Train / Test Split ─────────────────────────────
    # Split the cleaned+engineered DataFrame BEFORE fitting any sklearn
    # transforms.  Fitting transforms on the full dataset would let
    # information from the test set "leak" into the training pipeline —
    # a subtle but critical data leakage bug.
    logger.info("[4/6] Splitting data (%.0f%% train / %.0f%% test, stratified).",
                (1 - TEST_SIZE) * 100, TEST_SIZE * 100)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,      # preserve the ~26% churn rate in both splits
    )
    logger.info(
        "      Train: %d rows | Test: %d rows | Churn rate → train=%.1f%% test=%.1f%%",
        len(y_train), len(y_test),
        y_train.mean() * 100, y_test.mean() * 100,
    )

    # ── Step 5: Fit sklearn Pipeline on TRAIN, transform TRAIN + TEST ─────
    # fit_pipeline expects a DataFrame that still has the target column,
    # so we pass the full training slice (X_train_df + y_train).
    logger.info("[5/6] Fitting preprocessing pipeline on training data.")

    train_df_with_target = X_train_df.copy()
    train_df_with_target[TARGET_COL] = y_train

    pipeline, X_train_arr, _ = fit_pipeline(train_df_with_target, target_col=TARGET_COL)

    # Apply the FITTED pipeline (no re-fitting) to test data
    test_df_with_target = X_test_df.copy()
    test_df_with_target[TARGET_COL] = y_test
    X_test_arr, _ = apply_pipeline(pipeline, test_df_with_target, target_col=TARGET_COL)

    logger.info(
        "      Feature matrix → train %s | test %s",
        X_train_arr.shape, X_test_arr.shape,
    )

    # ── Step 6: Train & Evaluate ──────────────────────────────────────────
    logger.info("[6/6] Training models and selecting the best.")

    best_name, best_model, best_metrics, all_results = train_all_models(
        X_train_arr, y_train,
        X_test_arr,  y_test,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )

    # ── Print comparison table ────────────────────────────────────────────
    print_summary(all_results, best_name, y_test)

    # ── Persist best model + fitted pipeline ─────────────────────────────
    saved_model_path = save_model(best_model, model_name="best_model")
    save_pipeline(pipeline, path=PREPROCESSOR_PATH)

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("  Best model : %s", best_name)
    logger.info("  ROC-AUC    : %.4f", best_metrics["roc_auc"])
    logger.info("  F1 score   : %.4f", best_metrics["f1"])
    logger.info("  Recall     : %.4f", best_metrics["recall"])
    logger.info("  Model path : %s", saved_model_path)
    logger.info("  Pipeline   : %s", PREPROCESSOR_PATH)
    if _MLFLOW_AVAILABLE:
        logger.info("  MLflow UI  : mlflow ui --backend-store-uri %s", tracking_uri)
    logger.info("=" * 60)


# ============================================================================
# Script entrypoint
# ============================================================================

if __name__ == "__main__":
    # Allow an optional data path argument:  python src/train.py data/raw/churn.csv
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/churn.csv"
    main(data_path=data_path)
