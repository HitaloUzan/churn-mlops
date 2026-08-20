"""Data ingestion for the Telco Customer Churn dataset."""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn CSV as-is, no transformations applied."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Copy it from Downloads\\Datasets\\Telco Customer Churn\\ "
            "into data/raw/ first."
        )
    return pd.read_csv(path)
