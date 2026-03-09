"""
tests/test_preprocessing.py
----------------------------
Unit tests for src/preprocessing.py.

Covers both phases:
  Phase 1 (Pandas cleaning) : standardise_columns, drop_duplicates,
                               handle_missing_values, cast_types, clean
  Phase 2 (sklearn pipeline): build_pipeline, fit_pipeline,
                               apply_pipeline, save_pipeline, load_pipeline,
                               run_preprocessing

MLOps role: Run in CI on every push. If a transform breaks, the pipeline
            fails here rather than silently producing bad predictions.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    standardise_columns,
    drop_duplicates,
    handle_missing_values,
    cast_types,
    clean,
    get_column_types,
    build_pipeline,
    fit_pipeline,
    apply_pipeline,
    save_pipeline,
    load_pipeline,
    run_preprocessing,
)


# ---------------------------------------------------------------------------
# Shared fixture — a small Telco-like DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Minimal Telco-like raw DataFrame for testing."""
    return pd.DataFrame({
        "customerID":       ["001", "002", "003", "003"],  # "003" is duplicate
        "gender":           ["Male", "Female", "Male", "Male"],
        "SeniorCitizen":    [0, 1, 0, 0],
        "Partner":          ["Yes", "No", "Yes", "Yes"],
        "Dependents":       ["No", "No", "Yes", "Yes"],
        "tenure":           [12, None, 36, 36],
        "PhoneService":     ["Yes", "Yes", "No", "No"],
        "MultipleLines":    ["No", "Yes", "No phone service", "No phone service"],
        "InternetService":  ["DSL", "Fiber optic", "No", "No"],
        "OnlineSecurity":   ["Yes", "No", "No internet service", "No internet service"],
        "OnlineBackup":     ["No", "Yes", "No internet service", "No internet service"],
        "DeviceProtection": ["No", "No", "No internet service", "No internet service"],
        "TechSupport":      ["No", "No", "No internet service", "No internet service"],
        "StreamingTV":      ["No", "No", "No internet service", "No internet service"],
        "StreamingMovies":  ["No", "No", "No internet service", "No internet service"],
        "Contract":         ["Month-to-month", "One year", "Month-to-month", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes", "Yes"],
        "PaymentMethod":    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Bank transfer (automatic)"],
        "MonthlyCharges":   [55.0, 80.0, 25.0, 25.0],    # rows 2&3 are exact duplicates
        "TotalCharges":     ["660", "1920", "900", "900"],  # rows 2&3 are exact duplicates
        "Churn":            ["No", "Yes", "No", "No"],
    })


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    """Pre-cleaned DataFrame (output of clean())."""
    return clean(raw_df)


# =============================================================================
# Phase 1 — Pandas Cleaning
# =============================================================================

class TestStandardiseColumns:
    def test_lowercases_all_columns(self):
        df = pd.DataFrame({"CustomerID": [], "Monthly Charges": []})
        result = standardise_columns(df)
        assert "customerid" in result.columns
        assert "monthly_charges" in result.columns

    def test_replaces_hyphens_with_underscore(self):
        df = pd.DataFrame({"some-feature": []})
        result = standardise_columns(df)
        assert "some_feature" in result.columns

    def test_strips_leading_trailing_whitespace(self):
        df = pd.DataFrame({" tenure ": []})
        result = standardise_columns(df)
        assert "tenure" in result.columns

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"CustomerID": [1]})
        standardise_columns(df)
        assert "CustomerID" in df.columns  # original unchanged


class TestDropDuplicates:
    def test_removes_duplicate_rows(self, raw_df):
        df = standardise_columns(raw_df)
        result = drop_duplicates(df)
        assert len(result) == 3  # row "003" was duplicated

    def test_no_change_when_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = drop_duplicates(df)
        assert len(result) == 3


class TestHandleMissingValues:
    def test_numeric_nulls_filled(self):
        df = pd.DataFrame({"val": [1.0, None, 3.0]})
        result = handle_missing_values(df)
        assert result["val"].isnull().sum() == 0

    def test_categorical_nulls_filled(self):
        df = pd.DataFrame({"cat": ["a", None, "a"]})
        result = handle_missing_values(df)
        assert result["cat"].isnull().sum() == 0

    def test_high_missing_column_dropped(self):
        df = pd.DataFrame({
            "keep": [1, 2, 3],
            "drop_me": [None, None, None],   # 100% missing
        })
        result = handle_missing_values(df, drop_threshold=0.5)
        assert "drop_me" not in result.columns
        assert "keep" in result.columns

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"val": [1.0, None]})
        handle_missing_values(df)
        assert df["val"].isnull().sum() == 1  # original still has null


class TestCastTypes:
    def test_churn_yes_no_encoded_as_int(self):
        df = pd.DataFrame({"churn": ["Yes", "No", "yes", "no"]})
        result = cast_types(df)
        assert set(result["churn"].unique()) <= {0, 1}
        assert result["churn"].dtype == int

    def test_totalcharges_converted_to_float(self):
        df = pd.DataFrame({"totalcharges": ["100.0", "200.5", " "]})
        result = cast_types(df)
        assert pd.api.types.is_float_dtype(result["totalcharges"])
        # whitespace-only → NaN
        assert pd.isna(result["totalcharges"].iloc[2])

    def test_original_not_mutated(self):
        df = pd.DataFrame({"churn": ["Yes", "No"]})
        cast_types(df)
        # Check that values are unchanged; avoid dtype == object which breaks on
        # pandas 3.x StringDtype.
        assert set(df["churn"].tolist()) == {"Yes", "No"}


class TestClean:
    def test_returns_dataframe(self, raw_df):
        result = clean(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_nulls_in_output(self, raw_df):
        result = clean(raw_df)
        assert result.isnull().sum().sum() == 0

    def test_customerid_dropped(self, raw_df):
        result = clean(raw_df)
        assert "customerid" not in result.columns

    def test_column_names_lowercase(self, raw_df):
        result = clean(raw_df)
        assert all(c == c.lower() for c in result.columns)

    def test_duplicates_removed(self, raw_df):
        result = clean(raw_df)
        assert len(result) == 3   # one duplicate removed

    def test_churn_is_int(self, raw_df):
        result = clean(raw_df)
        assert result["churn"].dtype == int


# =============================================================================
# Phase 2 — Sklearn Pipeline
# =============================================================================

class TestGetColumnTypes:
    def test_returns_two_lists(self, clean_df):
        num, cat = get_column_types(clean_df)
        assert isinstance(num, list)
        assert isinstance(cat, list)

    def test_target_not_in_either_list(self, clean_df):
        num, cat = get_column_types(clean_df, target_col="churn")
        assert "churn" not in num
        assert "churn" not in cat

    def test_no_overlap_between_lists(self, clean_df):
        num, cat = get_column_types(clean_df)
        assert set(num).isdisjoint(set(cat))


class TestBuildPipeline:
    def test_returns_column_transformer(self, clean_df):
        from sklearn.compose import ColumnTransformer
        num, cat = get_column_types(clean_df)
        pipeline = build_pipeline(num, cat)
        assert hasattr(pipeline, "fit_transform")

    def test_has_num_and_cat_transformers(self, clean_df):
        num, cat = get_column_types(clean_df)
        pipeline = build_pipeline(num, cat)
        transformer_names = [name for name, _, _ in pipeline.transformers]
        assert "num" in transformer_names
        assert "cat" in transformer_names


class TestFitPipeline:
    def test_returns_three_values(self, clean_df):
        result = fit_pipeline(clean_df)
        assert len(result) == 3

    def test_X_is_2d_numpy_array(self, clean_df):
        _, X, _ = fit_pipeline(clean_df)
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2

    def test_y_is_1d_array(self, clean_df):
        _, _, y = fit_pipeline(clean_df)
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

    def test_X_row_count_matches_df(self, clean_df):
        _, X, _ = fit_pipeline(clean_df)
        assert X.shape[0] == len(clean_df)

    def test_raises_if_target_missing(self, clean_df):
        df_no_target = clean_df.drop(columns=["churn"])
        with pytest.raises(ValueError, match="not found"):
            fit_pipeline(df_no_target)


class TestApplyPipeline:
    def test_returns_same_shape_as_fit(self, clean_df):
        pipeline, X_train, _ = fit_pipeline(clean_df)
        X_test, y_test = apply_pipeline(pipeline, clean_df)
        assert X_test.shape == X_train.shape

    def test_y_returned_when_target_present(self, clean_df):
        pipeline, _, _ = fit_pipeline(clean_df)
        _, y = apply_pipeline(pipeline, clean_df)
        assert y is not None

    def test_y_none_when_target_absent(self, clean_df):
        pipeline, _, _ = fit_pipeline(clean_df)
        df_no_target = clean_df.drop(columns=["churn"])
        _, y = apply_pipeline(pipeline, df_no_target)
        assert y is None


class TestSaveLoadPipeline:
    def test_save_and_load_roundtrip(self, clean_df, tmp_path):
        pipeline, _, _ = fit_pipeline(clean_df)
        save_path = tmp_path / "pipeline.joblib"
        save_pipeline(pipeline, path=save_path)

        assert save_path.exists()

        loaded = load_pipeline(path=save_path)
        # Loaded pipeline should produce identical output
        X_original, _ = apply_pipeline(pipeline, clean_df)
        X_loaded,   _ = apply_pipeline(loaded,   clean_df)
        np.testing.assert_array_almost_equal(X_original, X_loaded)

    def test_load_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_pipeline(path=tmp_path / "nonexistent.joblib")


class TestRunPreprocessing:
    def test_returns_pipeline_X_y(self, raw_df):
        pipeline, X, y = run_preprocessing(raw_df, save_path=None)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

    def test_saves_pipeline_to_disk(self, raw_df, tmp_path):
        save_path = tmp_path / "preprocessor.joblib"
        run_preprocessing(raw_df, save_path=save_path)
        assert save_path.exists()

    def test_no_save_when_path_is_none(self, raw_df, tmp_path):
        # Should not raise, and no file should be created
        run_preprocessing(raw_df, save_path=None)
        assert not (tmp_path / "preprocessor.joblib").exists()
