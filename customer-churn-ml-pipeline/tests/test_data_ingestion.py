"""
tests/test_data_ingestion.py
-----------------------------
Unit tests for src/data_ingestion.py.

Tests cover:
  1. load_csv    — file found, file missing, empty file
  2. validate_schema — all columns present, missing columns
  3. log_data_summary — correct metadata extraction
  4. ingest      — full orchestrator happy path and error path

MLOps role: These run in CI on every push.
            If someone changes the CSV source or renames a column,
            the pipeline fails here — not silently in production.
"""

import sys
import os
import io
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_ingestion import load_csv, validate_schema, log_data_summary, ingest, REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path) -> str:
    """Write a minimal valid CSV to a temp file and return its path."""
    content = ",".join(REQUIRED_COLUMNS) + "\n"
    content += "001,Male,0,Yes,No,12,Yes,No,DSL,Yes,No,No,No,No,No,Month-to-month,Yes,Electronic check,55.0,660.0,No\n"
    content += "002,Female,1,No,No,24,Yes,No,Fiber optic,No,No,No,No,No,No,One year,No,Mailed check,75.0,1800.0,Yes\n"
    csv_file = tmp_path / "churn.csv"
    csv_file.write_text(content)
    return str(csv_file)


@pytest.fixture
def empty_csv(tmp_path) -> str:
    """Write a CSV with headers only (zero data rows)."""
    content = ",".join(REQUIRED_COLUMNS) + "\n"
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text(content)
    return str(csv_file)


@pytest.fixture
def sample_df(sample_csv) -> pd.DataFrame:
    """Return a loaded DataFrame from the sample CSV fixture."""
    return pd.read_csv(sample_csv)


# ---------------------------------------------------------------------------
# load_csv tests
# ---------------------------------------------------------------------------

class TestLoadCsv:
    def test_returns_dataframe(self, sample_csv):
        df = load_csv(sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self, sample_csv):
        df = load_csv(sample_csv)
        assert df.shape == (2, len(REQUIRED_COLUMNS))

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_csv(tmp_path / "nonexistent.csv")

    def test_empty_file_raises_value_error(self, empty_csv):
        with pytest.raises(ValueError, match="zero rows"):
            load_csv(empty_csv)

    def test_accepts_path_object(self, sample_csv):
        from pathlib import Path
        df = load_csv(Path(sample_csv))
        assert not df.empty

    def test_accepts_string_path(self, sample_csv):
        df = load_csv(str(sample_csv))
        assert not df.empty


# ---------------------------------------------------------------------------
# validate_schema tests
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_passes_with_all_columns(self, sample_df):
        # Should not raise
        validate_schema(sample_df)

    def test_raises_when_column_missing(self, sample_df):
        df_missing = sample_df.drop(columns=["Churn"])
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_schema(df_missing)

    def test_error_message_lists_missing_columns(self, sample_df):
        df_missing = sample_df.drop(columns=["Churn", "tenure"])
        with pytest.raises(ValueError) as exc_info:
            validate_schema(df_missing)
        assert "Churn" in str(exc_info.value)
        assert "tenure" in str(exc_info.value)

    def test_custom_required_columns(self, sample_df):
        # Should pass when we only require columns that exist
        validate_schema(sample_df, required_columns=["customerID", "Churn"])

    def test_empty_required_columns_always_passes(self, sample_df):
        # Passing an empty list means no requirements
        validate_schema(sample_df, required_columns=[])


# ---------------------------------------------------------------------------
# log_data_summary tests
# ---------------------------------------------------------------------------

class TestLogDataSummary:
    def test_returns_dict(self, sample_df):
        result = log_data_summary(sample_df)
        assert isinstance(result, dict)

    def test_correct_row_count(self, sample_df):
        result = log_data_summary(sample_df)
        assert result["n_rows"] == 2

    def test_correct_col_count(self, sample_df):
        result = log_data_summary(sample_df)
        assert result["n_cols"] == len(REQUIRED_COLUMNS)

    def test_churn_rate_is_float(self, sample_df):
        result = log_data_summary(sample_df)
        assert isinstance(result["churn_rate_pct"], float)

    def test_churn_rate_correct(self, sample_df):
        # 1 out of 2 customers churned → 50%
        result = log_data_summary(sample_df)
        assert result["churn_rate_pct"] == 50.0

    def test_no_missing_in_sample(self, sample_df):
        result = log_data_summary(sample_df)
        assert result["n_missing"] == 0


# ---------------------------------------------------------------------------
# ingest orchestrator tests
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_returns_dataframe(self, sample_csv):
        df = ingest(sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_ingest_schema_passes(self, sample_csv):
        # Should not raise — CSV has all required columns
        df = ingest(sample_csv, validate=True)
        assert not df.empty

    def test_ingest_skip_validation(self, tmp_path):
        # CSV with only 2 columns — would fail validation, but validate=False skips it
        minimal_csv = tmp_path / "minimal.csv"
        minimal_csv.write_text("col_a,col_b\n1,2\n3,4\n")
        df = ingest(str(minimal_csv), validate=False)
        assert df.shape == (2, 2)

    def test_ingest_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest(str(tmp_path / "missing.csv"))
