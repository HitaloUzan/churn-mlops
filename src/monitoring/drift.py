"""Data drift detection between a reference dataset (what the champion model
trained on) and a current dataset (simulating newly arrived production data).

Feature drift only — `Churn` is dropped before the comparison, since in a real
deployment the label for freshly arrived traffic usually isn't available yet;
what you actually monitor is whether the *input* distribution has shifted.

Trigger logic is importance-weighted, not a flat "% of columns drifted"
threshold (see docs/decisions.md ADR-005): a blanket 30%-of-columns rule
missed the exact drift injected in testing, because only 3 of 19 columns were
perturbed — 15.8%, under any reasonable blanket threshold, even though those
3 columns (Contract, tenure, MonthlyCharges) are the model's top SHAP
features from Project 1. Retraining triggers if EITHER the blanket share
crosses the threshold OR any critical feature drifts past its own threshold.
"""

from pathlib import Path

import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

from mlops.config import DRIFT_SHARE_THRESHOLD
from pipeline.clean import TARGET_COLUMN

# Top features by mean |SHAP value| from Project 1 (churn-pipeline docs/decisions.md
# / models/shap_top_features.json) that exist as raw input columns here — a drift
# monitor that only counts "% of columns" treats every feature as equally important,
# which these actually aren't.
CRITICAL_FEATURES = [
    "Contract",
    "tenure",
    "OnlineSecurity",
    "InternetService",
    "MonthlyCharges",
    "TotalCharges",
    "PaymentMethod",
]


def _is_drifted(method: str, value: float, threshold: float) -> bool:
    """p-value methods drift when value < threshold; distance methods drift
    when value > threshold — evidently's own convention per method family."""
    if "p_value" in method:
        return value < threshold
    return value > threshold


def compute_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_share_threshold: float = DRIFT_SHARE_THRESHOLD,
) -> dict:
    ref_features = reference_df.drop(columns=[TARGET_COLUMN], errors="ignore")
    cur_features = current_df.drop(columns=[TARGET_COLUMN], errors="ignore")

    ref_dataset = Dataset.from_pandas(ref_features)
    cur_dataset = Dataset.from_pandas(cur_features)

    report = Report(metrics=[DataDriftPreset(drift_share=drift_share_threshold)])
    snapshot = report.run(reference_data=ref_dataset, current_data=cur_dataset)
    result = snapshot.dict()

    n_columns = len(ref_features.columns)
    n_drifted_columns = 0
    per_column = {}

    for metric in result["metrics"]:
        name = metric["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            n_drifted_columns = int(metric["value"]["count"])
        elif name.startswith("ValueDrift(column="):
            column = name.split("column=", 1)[1].split(",")[0]
            method = name.split("method=", 1)[1].split(",")[0]
            threshold = float(name.split("threshold=", 1)[1].rstrip(")"))
            value = float(metric["value"])
            per_column[column] = {
                "value": value,
                "method": method,
                "threshold": threshold,
                "drifted": _is_drifted(method, value, threshold),
            }

    drift_share = n_drifted_columns / n_columns if n_columns else 0.0
    drifted_critical_features = [
        f for f in CRITICAL_FEATURES if per_column.get(f, {}).get("drifted")
    ]

    return {
        "drift_detected": (drift_share >= drift_share_threshold) or bool(drifted_critical_features),
        "trigger_reason": (
            "blanket_share" if drift_share >= drift_share_threshold
            else "critical_feature" if drifted_critical_features
            else "none"
        ),
        "drift_share": round(drift_share, 4),
        "drift_share_threshold": drift_share_threshold,
        "n_drifted_columns": n_drifted_columns,
        "n_columns": n_columns,
        "drifted_critical_features": drifted_critical_features,
        "per_column_drift": per_column,
        "snapshot": snapshot,
    }


def save_html_report(snapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
