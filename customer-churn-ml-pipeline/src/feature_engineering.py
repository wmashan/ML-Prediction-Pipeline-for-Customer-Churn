"""
feature_engineering.py
-----------------------
Creates new, model-ready columns from the cleaned Telco churn DataFrame.

Design principle
----------------
This module is PURELY Pandas — no sklearn, no fitting, no state.
It runs AFTER preprocessing.py (which handles cleaning + sklearn Pipeline)
and BEFORE train.py (which fits the model).

Why keep feature engineering separate from preprocessing?
  - Preprocessing = correctness (fix dtypes, fill nulls, standardise names).
  - Feature engineering = insight (encode domain knowledge as numeric signals).
  - Separation makes each layer independently testable and reusable.

Features added (all explainable to a business stakeholder)
----------------------------------------------------------
  tenure_group         : Bucket tenure into 5 lifecycle bands (0-1yr … 6yr+).
                         ML benefit: Tree models handle ordinal bands better than
                         a raw integer; also exposes non-linear churn patterns
                         (e.g. churn spikes in the first 12 months).

  avg_monthly_spend    : totalcharges / (tenure + 1).
                         Proxy for how much a customer pays per month even when
                         'totalcharges' and 'tenure' encode similar info.
                         ML benefit: Ratio features often help linear models.

  service_count        : Number of active value-added services (0-6).
                         Customers with many services are "stickier" — harder to
                         churn because switching cost is higher.

  contract_risk_score  : Numeric encoding of contract type (3=month-to-month,
                         2=one year, 1=two year). Month-to-month customers
                         can leave any time, making them highest-risk.

  is_digital_only      : 1 if Paperless Billing AND Electronic check payment.
                         These customers interact with Telco only digitally and
                         historically show higher churn rates.

MLOps note
----------
All functions are pure (input → output, no side-effects). They are safe to
call during both training and inference without refitting anything.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Service columns used for service_count feature
# ============================================================================

# The Telco dataset has six boolean "add-on" service columns.
# After preprocessing.py's standardise_columns(), names are lowercase.
_SERVICE_COLUMNS: List[str] = [
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
]

# Risk score mapping: higher number = more likely to churn
_CONTRACT_RISK: dict = {
    "month-to-month": 3,  # no lock-in → easiest to leave
    "one year":       2,  # mild lock-in
    "two year":       1,  # strong lock-in
}


# ============================================================================
# Individual feature functions
# ============================================================================

def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket 'tenure' (months) into five lifecycle bands.

    Bands
    -----
    "0-1yr"   :  0 – 11 months  (new customers, highest churn risk)
    "1-2yr"   : 12 – 23 months
    "2-4yr"   : 24 – 47 months
    "4-6yr"   : 48 – 71 months
    "6yr+"    : 72+ months       (veterans, lowest churn risk)

    Why a string label instead of a number?
      The sklearn OneHotEncoder in preprocessing.py will turn this into
      binary columns, which lets the model learn a separate coefficient
      for each band rather than forcing a linear tenure relationship.

    Requires column : 'tenure'
    New column      : 'tenure_group'  (string, one of the five labels above)
    """
    if "tenure" not in df.columns:
        logger.warning("'tenure' column missing — skipping add_tenure_group.")
        return df

    df = df.copy()
    bins   = [0,  12,  24,  48,  72,  np.inf]
    labels = ["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6yr+"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, right=False
    ).astype(str)

    logger.debug("Added 'tenure_group'. Distribution:\n%s",
                 df["tenure_group"].value_counts().to_string())
    return df


def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average monthly spend as a ratio feature.

    Formula:  avg_monthly_spend = totalcharges / (tenure + 1)

    Adding 1 to tenure avoids division-by-zero for brand-new customers
    (tenure = 0).

    Why this helps
    --------------
    'monthlycharges' captures the *current* plan price, but
    'avg_monthly_spend' reflects *historical* behaviour including plan
    changes, promotions, and overage charges.  A customer with high
    avg_monthly_spend relative to their currentmonthlycharges may have
    downgraded — a potential churn signal.

    Requires columns : 'totalcharges', 'tenure'
    New column       : 'avg_monthly_spend'  (float)
    """
    required = {"totalcharges", "tenure"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("Skipping add_avg_monthly_spend — missing: %s", missing)
        return df

    df = df.copy()
    df["avg_monthly_spend"] = df["totalcharges"] / (df["tenure"] + 1)

    logger.debug("Added 'avg_monthly_spend'. Stats:\n%s",
                 df["avg_monthly_spend"].describe().to_string())
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of active value-added services per customer (0–6).

    Services counted (any column that equals 'yes' case-insensitively):
      onlinesecurity, onlinebackup, deviceprotection,
      techsupport, streamingtv, streamingmovies

    Why this helps
    --------------
    Each extra service increases the switching cost for a customer.
    A single integer 'service_count' gives the model a compact
    "stickiness" signal that correlates negatively with churn.
    Without it, the model would need to combine six sparse OHE columns
    to discover the same pattern.

    Requires columns : any subset of the six service columns
    New column       : 'service_count'  (int, 0–6)
    """
    present = [c for c in _SERVICE_COLUMNS if c in df.columns]
    if not present:
        logger.warning("No service columns found — skipping add_service_count.")
        return df

    df = df.copy()
    # Normalise to lowercase string so 'Yes', 'YES', 'yes' all count
    service_flags = (
        df[present].astype(str).apply(lambda col: col.str.strip().str.lower())
        == "yes"
    ).astype(int)
    df["service_count"] = service_flags.sum(axis=1)

    logger.debug("Added 'service_count'. Distribution:\n%s",
                 df["service_count"].value_counts().sort_index().to_string())
    return df


def add_contract_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode contract type as an ordinal churn-risk score.

    Mapping
    -------
    month-to-month → 3  (can churn tomorrow — highest risk)
    one year       → 2  (ETF discourages churn for one year)
    two year       → 1  (strongest lock-in — lowest risk)
    unknown / NaN  → 2  (neutral default)

    Why an ordinal number instead of OHE?
      Contract type has a natural order (shorter = riskier), so a single
      ordinal integer lets linear models learn a monotone relationship.
      OHE would treat the three values as independent categories and lose
      the ordering information.

    Requires column : 'contract'
    New column      : 'contract_risk_score'  (int, 1–3)
    """
    if "contract" not in df.columns:
        logger.warning("'contract' column missing — skipping add_contract_risk_score.")
        return df

    df = df.copy()
    normalised = df["contract"].astype(str).str.strip().str.lower()
    df["contract_risk_score"] = normalised.map(_CONTRACT_RISK).fillna(2).astype(int)

    logger.debug("Added 'contract_risk_score'. Distribution:\n%s",
                 df["contract_risk_score"].value_counts().sort_index().to_string())
    return df


def add_is_digital_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag customers who use only digital / self-service channels.

    Rule: is_digital_only = 1  if  PaperlessBilling == 'yes'
                                   AND PaymentMethod == 'electronic check'

    Why this helps
    --------------
    Digital-only customers have fewer human touch-points with the Telco
    (no paper bills, no direct-debit conversation with the bank).
    Research on this dataset shows this group churns at ~10 pp above average.
    A single binary flag is easier for the model to exploit than two OHE
    dummies that it must implicitly combine.

    Requires columns : 'paperlessbilling', 'paymentmethod'
    New column       : 'is_digital_only'  (int, 0 or 1)
    """
    required = {"paperlessbilling", "paymentmethod"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("Skipping add_is_digital_only — missing: %s", missing)
        return df

    df = df.copy()
    paperless = df["paperlessbilling"].astype(str).str.strip().str.lower() == "yes"
    echeck    = df["paymentmethod"].astype(str).str.strip().str.lower() == "electronic check"
    df["is_digital_only"] = (paperless & echeck).astype(int)

    logger.debug("Added 'is_digital_only'. Value counts:\n%s",
                 df["is_digital_only"].value_counts().to_string())
    return df


# ============================================================================
# Orchestrator — runs all feature functions in sequence
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps and return the enriched DataFrame.

    Calling order matters: later features may use columns added by earlier ones,
    but in practice these five are independent of each other.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from preprocessing.clean() — column names must be
        lowercase with underscores (already done by standardise_columns).

    Returns
    -------
    pd.DataFrame
        Same rows as input, with up to 5 extra columns:
          tenure_group, avg_monthly_spend, service_count,
          contract_risk_score, is_digital_only.

    Usage in train.py
    -----------------
        from src.preprocessing      import clean
        from src.feature_engineering import engineer_features

        df = clean(raw_df)
        df = engineer_features(df)        # ← add domain features
        pipeline, X, y = fit_pipeline(df)  # ← fit sklearn transforms
    """
    logger.info("Starting feature engineering on DataFrame with shape %s.", df.shape)

    df = add_tenure_group(df)
    df = add_avg_monthly_spend(df)
    df = add_service_count(df)
    df = add_contract_risk_score(df)
    df = add_is_digital_only(df)

    new_cols = [
        "tenure_group", "avg_monthly_spend",
        "service_count", "contract_risk_score", "is_digital_only",
    ]
    created = [c for c in new_cols if c in df.columns]
    logger.info("Feature engineering complete. New columns: %s", created)
    return df


# ============================================================================
# Standalone smoke-test  (python src/feature_engineering.py)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    sample = pd.DataFrame({
        "customerid":       ["001", "002", "003"],
        "tenure":           [2,     24,    60],
        "totalcharges":     [110.0, 1920.0, 5400.0],
        "monthlycharges":   [55.0,  80.0,   90.0],
        "contract":         ["Month-to-month", "One year", "Two year"],
        "paperlessbilling": ["Yes", "No",  "Yes"],
        "paymentmethod":    ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "onlinesecurity":   ["No",  "Yes", "Yes"],
        "onlinebackup":     ["No",  "Yes", "Yes"],
        "deviceprotection": ["No",  "No",  "Yes"],
        "techsupport":      ["No",  "No",  "Yes"],
        "streamingtv":      ["No",  "No",  "Yes"],
        "streamingmovies":  ["No",  "No",  "Yes"],
        "churn":            [1, 0, 0],
    })

    enriched = engineer_features(sample)
    print("\nEnriched DataFrame:")
    print(enriched[[
        "customerid", "tenure", "tenure_group",
        "avg_monthly_spend", "service_count",
        "contract_risk_score", "is_digital_only",
    ]].to_string(index=False))
    print("\nAll columns:", enriched.columns.tolist())
