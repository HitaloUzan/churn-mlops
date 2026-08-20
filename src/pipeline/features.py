"""Feature engineering for the Telco Customer Churn dataset.

Adds business-meaningful features on top of the raw columns, then defines the
preprocessing (encoding/scaling) applied before the model. Both steps are
wrapped into the single sklearn Pipeline serialized by src/model/train.py, so
the API only ever has to pass raw, cleaned feature dicts in.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADDON_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "tenure_years",
    "avg_monthly_spend",
    "num_addon_services",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
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
]


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive tenure/spend/adoption features that plain columns don't capture."""
    df = df.copy()

    df["tenure_years"] = df["tenure"] / 12

    # avg_monthly_spend differs from MonthlyCharges for customers who changed
    # plans mid-contract; for tenure == 0 (brand-new) fall back to MonthlyCharges.
    df["avg_monthly_spend"] = (df["TotalCharges"] / df["tenure"]).where(
        df["tenure"] > 0, df["MonthlyCharges"]
    )

    df["num_addon_services"] = (df[ADDON_SERVICE_COLUMNS] == "Yes").sum(axis=1)

    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_feature_pipeline() -> Pipeline:
    """Business feature engineering only — precedes the model-specific preprocessor."""
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline(
        steps=[
            ("business_features", FunctionTransformer(add_business_features)),
        ]
    )
