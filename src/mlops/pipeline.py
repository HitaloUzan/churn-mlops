"""The retraining pipeline — this is what GitHub Actions runs on a schedule
(or manual dispatch). Checks drift; retrains and attempts promotion only if
drift crosses the threshold (see docs/decisions.md ADR-005 and ADR-006).

Run with: python -m src.mlops.pipeline [--no-drift]
  --no-drift  use undrifted current data (demonstrates the "nothing to do"
              path — see README for why both paths matter for the demo)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mlops.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, PROMOTION_METRIC
from mlops.data_splits import make_splits
from mlops.governance import promote_if_better
from mlops.train import log_and_register, train_and_evaluate
from monitoring.drift import compute_drift, save_html_report
from monitoring.simulate_drift import inject_drift

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
MODELS_DIR = ROOT / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-drift",
        action="store_true",
        help="Use undrifted current_raw instead of the injected-drift simulation",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    print("Loading fixed splits...")
    splits = make_splits()
    reference, eval_holdout, current_raw = splits["reference"], splits["eval_holdout"], splits["current_raw"]

    current = current_raw if args.no_drift else inject_drift(current_raw)
    print(f"Using {'undrifted' if args.no_drift else 'drift-injected'} current data ({len(current)} rows)")

    print("Computing drift report (reference vs. current)...")
    drift_result = compute_drift(reference, current)
    print(
        f"  drift_detected={drift_result['drift_detected']} "
        f"reason={drift_result['trigger_reason']} "
        f"share={drift_result['drift_share']} "
        f"critical_features={drift_result['drifted_critical_features']}"
    )

    DOCS_DIR.mkdir(exist_ok=True)
    save_html_report(drift_result["snapshot"], DOCS_DIR / "drift_report.html")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "drift_mode": "undrifted" if args.no_drift else "drift_injected",
        "drift_detected": drift_result["drift_detected"],
        "trigger_reason": drift_result["trigger_reason"],
        "drift_share": drift_result["drift_share"],
        "n_drifted_columns": drift_result["n_drifted_columns"],
        "n_columns": drift_result["n_columns"],
        "drifted_critical_features": drift_result["drifted_critical_features"],
        "retrained": False,
        "promotion": None,
    }

    if drift_result["drift_detected"]:
        print("Drift crossed the trigger threshold — retraining candidate on reference + current...")
        import pandas as pd

        combined_train = pd.concat([reference, current], ignore_index=True)

        t0 = time.time()
        candidate_pipeline, candidate_metrics = train_and_evaluate(combined_train, eval_holdout)
        train_seconds = round(time.time() - t0, 2)
        print(f"  candidate metrics on eval_holdout: {candidate_metrics}")

        run_id, candidate_version = log_and_register(
            candidate_pipeline,
            candidate_metrics,
            run_name="retrain-triggered",
            extra_params={
                "train_seconds": train_seconds,
                "trigger_reason": drift_result["trigger_reason"],
                "drift_share": drift_result["drift_share"],
            },
        )
        print(f"Registered candidate v{candidate_version} (run {run_id})")

        decision = promote_if_better(client, candidate_version, candidate_metrics, PROMOTION_METRIC)
        print(f"Governance decision: {decision}")

        summary["retrained"] = True
        summary["candidate_version"] = candidate_version
        summary["candidate_metrics"] = candidate_metrics
        summary["promotion"] = decision
    else:
        print("Drift below threshold — no retraining needed.")

    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "pipeline_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary to {MODELS_DIR / 'pipeline_run_summary.json'}")


if __name__ == "__main__":
    main()
