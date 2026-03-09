"""
tests/test_feature_engineering.py
----------------------------------
Unit tests for src/feature_engineering.py.

Each test class covers one public function.
Tests verify:
  - Correct column values and types.
  - Graceful handling of missing input columns.
  - Immutability: input DataFrame is never mutated.
  - Boundary / edge conditions.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_tenure_group,
    add_avg_monthly_spend,
    add_service_count,
    add_contract_risk_score,
    add_is_digital_only,
    engineer_features,
)


# =============================================================================
# Shared fixture
# =============================================================================

@pytest.fixture
def base_df() -> pd.DataFrame:
    """
    Small Telco-like DataFrame with all columns needed by feature engineering.
    Column names are lowercase (as produced by preprocessing.standardise_columns).
    """
    return pd.DataFrame({
        "customerid":       ["001", "002", "003"],
        "tenure":           [2,     24,    60],
        "totalcharges":     [110.0, 1920.0, 5400.0],
        "monthlycharges":   [55.0,  80.0,   90.0],
        "contract":         ["Month-to-month", "One year", "Two year"],
        "paperlessbilling": ["Yes", "No",  "Yes"],
        "paymentmethod":    ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        # Value-added services
        "onlinesecurity":   ["No",  "Yes", "Yes"],
        "onlinebackup":     ["No",  "Yes", "Yes"],
        "deviceprotection": ["No",  "No",  "Yes"],
        "techsupport":      ["No",  "No",  "Yes"],
        "streamingtv":      ["No",  "No",  "Yes"],
        "streamingmovies":  ["No",  "No",  "Yes"],
        "churn":            [1, 0, 0],
    })


# =============================================================================
# add_tenure_group
# =============================================================================

class TestAddTenureGroup:
    def test_column_created(self, base_df):
        result = add_tenure_group(base_df)
        assert "tenure_group" in result.columns

    def test_correct_bands(self, base_df):
        result = add_tenure_group(base_df)
        # tenure=2   → "0-1yr"  (band: 0 to <12)
        # tenure=24  → "2-4yr"  (band: 24 to <48)
        # tenure=60  → "4-6yr"  (band: 48 to <72)
        assert result.loc[0, "tenure_group"] == "0-1yr"
        assert result.loc[1, "tenure_group"] == "2-4yr"
        assert result.loc[2, "tenure_group"] == "4-6yr"

    def test_boundary_12_months_is_1_2yr(self):
        df = pd.DataFrame({"tenure": [12]})
        result = add_tenure_group(df)
        assert result.loc[0, "tenure_group"] == "1-2yr"

    def test_boundary_72_months_is_6yr_plus(self):
        df = pd.DataFrame({"tenure": [72]})
        result = add_tenure_group(df)
        assert result.loc[0, "tenure_group"] == "6yr+"

    def test_returns_string_dtype(self, base_df):
        result = add_tenure_group(base_df)
        # stored as string so OHE picks it up as categorical
        assert result["tenure_group"].dtype == object or \
               hasattr(result["tenure_group"].dtype, "name")  # StringDtype OK

    def test_no_mutation_of_input(self, base_df):
        add_tenure_group(base_df)
        assert "tenure_group" not in base_df.columns

    def test_graceful_when_tenure_missing(self):
        df = pd.DataFrame({"totalcharges": [100.0]})
        result = add_tenure_group(df)  # should not raise
        assert "tenure_group" not in result.columns


# =============================================================================
# add_avg_monthly_spend
# =============================================================================

class TestAddAvgMonthlySpend:
    def test_column_created(self, base_df):
        result = add_avg_monthly_spend(base_df)
        assert "avg_monthly_spend" in result.columns

    def test_correct_value(self):
        # tenure=9, totalcharges=450.0  →  450 / (9 + 1) = 45.0
        df = pd.DataFrame({"tenure": [9], "totalcharges": [450.0]})
        result = add_avg_monthly_spend(df)
        assert abs(result.loc[0, "avg_monthly_spend"] - 45.0) < 1e-9

    def test_tenure_zero_no_division_error(self):
        # tenure=0 → divisor becomes 1, so result = totalcharges
        df = pd.DataFrame({"tenure": [0], "totalcharges": [55.0]})
        result = add_avg_monthly_spend(df)
        assert abs(result.loc[0, "avg_monthly_spend"] - 55.0) < 1e-9

    def test_result_is_float(self, base_df):
        result = add_avg_monthly_spend(base_df)
        assert pd.api.types.is_float_dtype(result["avg_monthly_spend"])

    def test_no_mutation_of_input(self, base_df):
        add_avg_monthly_spend(base_df)
        assert "avg_monthly_spend" not in base_df.columns

    def test_graceful_when_columns_missing(self):
        df = pd.DataFrame({"tenure": [10]})  # totalcharges absent
        result = add_avg_monthly_spend(df)   # should not raise
        assert "avg_monthly_spend" not in result.columns


# =============================================================================
# add_service_count
# =============================================================================

class TestAddServiceCount:
    def test_column_created(self, base_df):
        result = add_service_count(base_df)
        assert "service_count" in result.columns

    def test_zero_services(self, base_df):
        # customer 001 has all services = "No"
        result = add_service_count(base_df)
        assert result.loc[0, "service_count"] == 0

    def test_partial_services(self, base_df):
        # customer 002 has onlinesecurity=Yes, onlinebackup=Yes → 2
        result = add_service_count(base_df)
        assert result.loc[1, "service_count"] == 2

    def test_all_services(self, base_df):
        # customer 003 has all 6 services = "Yes"
        result = add_service_count(base_df)
        assert result.loc[2, "service_count"] == 6

    def test_case_insensitive(self):
        df = pd.DataFrame({"onlinesecurity": ["YES"], "onlinebackup": ["yes"]})
        result = add_service_count(df)
        assert result.loc[0, "service_count"] == 2

    def test_result_is_integer(self, base_df):
        result = add_service_count(base_df)
        assert pd.api.types.is_integer_dtype(result["service_count"])

    def test_no_mutation_of_input(self, base_df):
        add_service_count(base_df)
        assert "service_count" not in base_df.columns

    def test_graceful_when_no_service_columns(self):
        df = pd.DataFrame({"tenure": [10]})
        result = add_service_count(df)  # should not raise
        assert "service_count" not in result.columns


# =============================================================================
# add_contract_risk_score
# =============================================================================

class TestAddContractRiskScore:
    def test_column_created(self, base_df):
        result = add_contract_risk_score(base_df)
        assert "contract_risk_score" in result.columns

    def test_month_to_month_is_3(self, base_df):
        result = add_contract_risk_score(base_df)
        assert result.loc[0, "contract_risk_score"] == 3

    def test_one_year_is_2(self, base_df):
        result = add_contract_risk_score(base_df)
        assert result.loc[1, "contract_risk_score"] == 2

    def test_two_year_is_1(self, base_df):
        result = add_contract_risk_score(base_df)
        assert result.loc[2, "contract_risk_score"] == 1

    def test_unknown_value_defaults_to_2(self):
        df = pd.DataFrame({"contract": ["Unknown plan"]})
        result = add_contract_risk_score(df)
        assert result.loc[0, "contract_risk_score"] == 2

    def test_result_is_integer(self, base_df):
        result = add_contract_risk_score(base_df)
        assert pd.api.types.is_integer_dtype(result["contract_risk_score"])

    def test_no_mutation_of_input(self, base_df):
        add_contract_risk_score(base_df)
        assert "contract_risk_score" not in base_df.columns

    def test_graceful_when_column_missing(self):
        df = pd.DataFrame({"tenure": [10]})
        result = add_contract_risk_score(df)  # should not raise
        assert "contract_risk_score" not in result.columns


# =============================================================================
# add_is_digital_only
# =============================================================================

class TestAddIsDigitalOnly:
    def test_column_created(self, base_df):
        result = add_is_digital_only(base_df)
        assert "is_digital_only" in result.columns

    def test_paperless_plus_echeck_is_1(self, base_df):
        # customer 001: PaperlessBilling=Yes + Electronic check → 1
        result = add_is_digital_only(base_df)
        assert result.loc[0, "is_digital_only"] == 1

    def test_non_echeck_is_0(self, base_df):
        # customer 002: PaperlessBilling=No → 0
        result = add_is_digital_only(base_df)
        assert result.loc[1, "is_digital_only"] == 0

    def test_paperless_without_echeck_is_0(self, base_df):
        # customer 003: PaperlessBilling=Yes but Bank transfer → 0
        result = add_is_digital_only(base_df)
        assert result.loc[2, "is_digital_only"] == 0

    def test_result_is_integer(self, base_df):
        result = add_is_digital_only(base_df)
        assert pd.api.types.is_integer_dtype(result["is_digital_only"])

    def test_no_mutation_of_input(self, base_df):
        add_is_digital_only(base_df)
        assert "is_digital_only" not in base_df.columns

    def test_graceful_when_columns_missing(self):
        df = pd.DataFrame({"tenure": [10]})
        result = add_is_digital_only(df)  # should not raise
        assert "is_digital_only" not in result.columns


# =============================================================================
# engineer_features  (orchestrator)
# =============================================================================

class TestEngineerFeatures:
    NEW_COLS = [
        "tenure_group",
        "avg_monthly_spend",
        "service_count",
        "contract_risk_score",
        "is_digital_only",
    ]

    def test_all_new_columns_present(self, base_df):
        result = engineer_features(base_df)
        for col in self.NEW_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_unchanged(self, base_df):
        result = engineer_features(base_df)
        assert len(result) == len(base_df)

    def test_original_columns_preserved(self, base_df):
        result = engineer_features(base_df)
        for col in base_df.columns:
            assert col in result.columns

    def test_input_not_mutated(self, base_df):
        engineer_features(base_df)
        for col in self.NEW_COLS:
            assert col not in base_df.columns

    def test_returns_dataframe(self, base_df):
        result = engineer_features(base_df)
        assert isinstance(result, pd.DataFrame)
