"""Shared config across training, drift monitoring, and the retraining pipeline.

MLflow aliases, not stages (see docs/decisions.md ADR-002): `set_registered_model_alias`
is MLflow's current recommended promotion mechanism — `transition_model_version_stage`
still works but is legacy. `champion` = the model actually serving predictions
(what the roadmap calls "production"); `challenger` = a newly trained candidate
that hasn't beaten the champion yet (what the roadmap calls "staging").
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "churn-retraining"
MODEL_NAME = "churn-classifier"

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

PROMOTION_METRIC = "recall"  # matches Project 1's priority metric (false negatives cost more)

DRIFT_SHARE_THRESHOLD = 0.3  # fraction of columns that must show significant drift to trigger retraining

RANDOM_STATE = 42
EVAL_HOLDOUT_FRAC = 0.2  # fixed, never trained on — the one constant yardstick both models are scored on
CURRENT_FRAC_OF_POOL = 0.3  # of the remaining pool, the share simulated as "post-deploy" data
