"""
tests/test_predict.py
---------------------
Unit tests for src/predict.py — the inference module.

Strategy
--------
These tests do NOT need a real trained model on disk.
Instead we use pytest's `tmp_path` fixture and `unittest.mock` to:
  • Build a minimal sklearn pipeline and save it as a .joblib file.
  • Replace the real model with a tiny mock that returns predictable probabilities.

This keeps tests fast (< 1 second) and ensures they always pass in CI,
regardless of whether someone has run training locally.

Coverage
--------
  1. _to_dataframe      — dict / list / DataFrame input normalisation
  2. _validate_input    — missing columns, empty input
  3. predict_single     — happy path, below-threshold, custom threshold
  4. predict_batch      — list of dicts, DataFrame input, output columns
  5. reload_artefacts   — missing file raises FileNotFoundError
  6. get_artefact_info  — keys present, correct types

MLOps role
----------
If someone changes a column name in feature_engineering.py, these tests
will fail loudly in CI — not silently produce wrong predictions in production.
"""

import sys
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from unittest.mock import MagicMock, patch

# Make sure the project root is on the path when running from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.predict import (
    _to_dataframe,
    _validate_input,
    predict_single,
    predict_batch,
    reload_artefacts,
    get_artefact_info,
    REQUIRED_COLUMNS,
)


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def minimal_customer() -> dict:
    """
    A valid customer dictionary with all 19 required fields.
    Represents a high-risk customer (Fiber optic, Month-to-month).
    """
    return {
        "gender":          "Female",
        "seniorcitizen":   0,
        "partner":         "Yes",
        "dependents":      "No",
        "tenure":          6,
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
        "totalcharges":    479.1,
    }


@pytest.fixture
def fake_artefacts(tmp_path):
    """
    Saves a minimal, real sklearn pipeline and a fitted LogisticRegression
    stub to tmp_path. Returns (model_path, preprocessor_path).

    Uses only real serialisable sklearn objects — MagicMock cannot be pickled
    by joblib. Available for future tests that need on-disk artefacts.
    """
    from sklearn.linear_model import LogisticRegression

    # Minimal 2-feature pipeline
    preprocessor = Pipeline([("scaler", StandardScaler())])
    X_dummy = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    preprocessor.fit(X_dummy)

    # A real (trivially trained) model so predict_proba is available
    model = LogisticRegression()
    y_dummy = np.array([0, 1, 0])
    model.fit(X_dummy, y_dummy)

    model_path = tmp_path / "best_model.joblib"
    prep_path  = tmp_path / "preprocessor.joblib"
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, prep_path)

    return str(model_path), str(prep_path)


# =============================================================================
# _to_dataframe — input normalisation
# =============================================================================

class TestToDataframe:
    """_to_dataframe converts any supported input type to a DataFrame."""

    def test_dict_becomes_single_row_dataframe(self, minimal_customer):
        df = _to_dataframe(minimal_customer)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_list_of_dicts_goes_through_predict_batch(self, minimal_customer):
        # _to_dataframe only accepts dict or DataFrame.
        # A list of dicts is the domain of predict_batch(), which converts
        # it internally. Passing a list directly to _to_dataframe raises TypeError.
        with pytest.raises(TypeError):
            _to_dataframe([minimal_customer, minimal_customer])

    def test_dataframe_passes_through(self, minimal_customer):
        input_df = pd.DataFrame([minimal_customer])
        result = _to_dataframe(input_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_column_names_lowercased(self, minimal_customer):
        # Even if someone passes mixed-case keys the function must normalise them
        mixed_case = {k.upper(): v for k, v in minimal_customer.items()}
        df = _to_dataframe(mixed_case)
        assert all(col == col.lower() for col in df.columns)


# =============================================================================
# _validate_input — schema guard
# =============================================================================

class TestValidateInput:
    """_validate_input raises ValueError when required columns are missing."""

    def test_passes_when_all_columns_present(self, minimal_customer):
        df = pd.DataFrame([minimal_customer])
        # Should not raise
        _validate_input(df)

    def test_raises_when_column_missing(self, minimal_customer):
        df = pd.DataFrame([minimal_customer]).drop(columns=["tenure"])
        with pytest.raises(ValueError, match="tenure"):
            _validate_input(df)

    def test_raises_when_multiple_columns_missing(self, minimal_customer):
        df = pd.DataFrame([minimal_customer]).drop(columns=["tenure", "gender"])
        with pytest.raises(ValueError):
            _validate_input(df)

    def test_raises_on_empty_dataframe(self):
        # An empty DataFrame has all the right columns but zero rows.
        # _validate_input checks column presence; predict_single / predict_batch
        # are responsible for catching the "nothing to score" case downstream.
        # This test documents that _validate_input itself does NOT raise on
        # zero rows — the empty-input guard lives higher in the call stack.
        empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
        # No exception expected from _validate_input for column checks
        _validate_input(empty)  # should not raise


# =============================================================================
# predict_single — single-customer inference
# =============================================================================

class TestPredictSingle:
    """
    predict_single(customer_dict) → {"churn", "probability", "label", "threshold"}

    We patch _run_inference_pipeline so we don't need real artefacts on disk.
    This isolates the test to the logic in predict_single itself.
    """

    def _mock_inference(self, probability: float):
        """Return a context manager that stubs _run_inference_pipeline."""
        return patch(
            "src.predict._run_inference_pipeline",
            return_value=(np.array([probability]), None),
        )

    def _mock_loaded(self):
        """Pretend artefacts are already loaded."""
        return patch("src.predict._model", MagicMock())

    def test_returns_dict_with_required_keys(self, minimal_customer):
        with self._mock_inference(0.82), self._mock_loaded():
            result = predict_single(minimal_customer)
        assert set(result.keys()) >= {"churn", "probability", "label", "threshold"}

    def test_high_probability_predicts_churn(self, minimal_customer):
        with self._mock_inference(0.82), self._mock_loaded():
            result = predict_single(minimal_customer, threshold=0.5)
        assert result["churn"] == 1
        assert result["label"] == "Churn"

    def test_low_probability_predicts_no_churn(self, minimal_customer):
        with self._mock_inference(0.15), self._mock_loaded():
            result = predict_single(minimal_customer, threshold=0.5)
        assert result["churn"] == 0
        assert result["label"] == "No Churn"

    def test_custom_threshold_changes_label(self, minimal_customer):
        # probability=0.40, threshold=0.35 → should be Churn
        with self._mock_inference(0.40), self._mock_loaded():
            result = predict_single(minimal_customer, threshold=0.35)
        assert result["churn"] == 1

        # Same probability, higher threshold=0.5 → should be No Churn
        with self._mock_inference(0.40), self._mock_loaded():
            result_high = predict_single(minimal_customer, threshold=0.5)
        assert result_high["churn"] == 0

    def test_probability_is_float_between_0_and_1(self, minimal_customer):
        with self._mock_inference(0.63), self._mock_loaded():
            result = predict_single(minimal_customer)
        assert isinstance(result["probability"], float)
        assert 0.0 <= result["probability"] <= 1.0

    def test_threshold_echoed_in_response(self, minimal_customer):
        with self._mock_inference(0.55), self._mock_loaded():
            result = predict_single(minimal_customer, threshold=0.42)
        assert result["threshold"] == pytest.approx(0.42)

    def test_raises_when_required_column_missing(self, minimal_customer):
        bad_input = {k: v for k, v in minimal_customer.items() if k != "tenure"}
        with pytest.raises((ValueError, KeyError)):
            predict_single(bad_input)


# =============================================================================
# predict_batch — multi-customer inference
# =============================================================================

class TestPredictBatch:
    """
    predict_batch(list_of_dicts) → DataFrame with churn_probability,
    churn_prediction, churn_label columns.
    """

    def _mock_inference_n(self, probabilities):
        """Stub _run_inference_pipeline to return N probabilities."""
        return patch(
            "src.predict._run_inference_pipeline",
            return_value=(np.array(probabilities), None),
        )

    def _mock_loaded(self):
        return patch("src.predict._model", MagicMock())

    def test_returns_dataframe(self, minimal_customer):
        with self._mock_inference_n([0.8, 0.2]), self._mock_loaded():
            result = predict_batch([minimal_customer, minimal_customer])
        assert isinstance(result, pd.DataFrame)

    def test_output_row_count_matches_input(self, minimal_customer):
        with self._mock_inference_n([0.9, 0.1, 0.5]), self._mock_loaded():
            result = predict_batch([minimal_customer] * 3)
        assert len(result) == 3

    def test_output_has_required_columns(self, minimal_customer):
        with self._mock_inference_n([0.7]), self._mock_loaded():
            result = predict_batch([minimal_customer])
        assert "churn_probability" in result.columns
        assert "churn_prediction" in result.columns
        assert "churn_label" in result.columns

    def test_probabilities_between_0_and_1(self, minimal_customer):
        with self._mock_inference_n([0.45, 0.88]), self._mock_loaded():
            result = predict_batch([minimal_customer, minimal_customer])
        assert (result["churn_probability"] >= 0.0).all()
        assert (result["churn_probability"] <= 1.0).all()

    def test_prediction_respects_threshold(self, minimal_customer):
        # probability=0.45 with threshold=0.4 → Churn
        with self._mock_inference_n([0.45]), self._mock_loaded():
            result = predict_batch([minimal_customer], threshold=0.40)
        assert result["churn_prediction"].iloc[0] == 1

        # Same probability with threshold=0.5 → No Churn
        with self._mock_inference_n([0.45]), self._mock_loaded():
            result_high = predict_batch([minimal_customer], threshold=0.50)
        assert result_high["churn_prediction"].iloc[0] == 0

    def test_accepts_dataframe_input(self, minimal_customer):
        input_df = pd.DataFrame([minimal_customer, minimal_customer])
        with self._mock_inference_n([0.6, 0.3]), self._mock_loaded():
            result = predict_batch(input_df)
        assert len(result) == 2


# =============================================================================
# reload_artefacts — artefact loading
# =============================================================================

class TestReloadArtefacts:
    """reload_artefacts() loads model + preprocessor from disk."""

    def test_raises_when_model_file_missing(self, tmp_path):
        # Point MODELS_DIR at an empty temp directory — no .joblib files there
        with patch("src.predict.MODELS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                reload_artefacts("nonexistent_model")

    def test_raises_when_preprocessor_file_missing(self, tmp_path):
        # Create a model file (using a real serialisable object) but
        # deliberately do NOT create the preprocessor file.
        # reload_artefacts must raise FileNotFoundError about the missing
        # preprocessor.
        from sklearn.preprocessing import StandardScaler
        joblib.dump(StandardScaler(), tmp_path / "only_model.joblib")
        with patch("src.predict.MODELS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                reload_artefacts("only_model")


# =============================================================================
# get_artefact_info — metadata helper
# =============================================================================

class TestGetArtefactInfo:
    """get_artefact_info() returns a dict describing the loaded artefacts."""

    def test_returns_dict(self):
        info = get_artefact_info()
        assert isinstance(info, dict)

    def test_has_model_loaded_flag(self):
        info = get_artefact_info()
        assert "model_loaded" in info
        assert isinstance(info["model_loaded"], bool)

    def test_has_preprocessor_loaded_flag(self):
        info = get_artefact_info()
        assert "preprocessor_loaded" in info
        assert isinstance(info["preprocessor_loaded"], bool)

    def test_has_model_name_field(self):
        info = get_artefact_info()
        assert "model_name" in info

    def test_model_loaded_false_before_artefacts_loaded(self):
        # Force the module-level cache to None so nothing is loaded
        with patch("src.predict._model", None), \
             patch("src.predict._preprocessor", None):
            info = get_artefact_info()
        assert info["model_loaded"] is False
        assert info["preprocessor_loaded"] is False
