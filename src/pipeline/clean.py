"""Cleaning steps for the Telco Customer Churn dataset.

Known data quality issue (documented in docs/decisions.md): 11 customers with
tenure == 0 have TotalCharges stored as an empty string instead of 0 — these are
brand-new customers who haven't been billed yet, not missing data.
"""

import pandas as pd

TARGET_COLUMN = "Churn"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    df = df.drop_duplicates(subset="customerID")
    df = df.set_index("customerID")

    return df
