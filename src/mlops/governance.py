"""Promotion governance: a retrained candidate only becomes champion if it
beats the current champion on the shared eval_holdout — retraining happening
is not, by itself, a reason to promote (see docs/decisions.md ADR-006).
"""

import mlflow
from mlflow.tracking import MlflowClient

from mlops.config import CHALLENGER_ALIAS, CHAMPION_ALIAS, MODEL_NAME, PROMOTION_METRIC


def get_champion(client: MlflowClient) -> tuple[str, dict] | tuple[None, None]:
    """Returns (version, metrics_dict) for the current champion, or (None, None)
    if no champion alias exists yet."""
    try:
        model_version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None, None
    run = client.get_run(model_version.run_id)
    return model_version.version, dict(run.data.metrics)


def promote_if_better(
    client: MlflowClient,
    candidate_version: str,
    candidate_metrics: dict,
    metric_key: str = PROMOTION_METRIC,
) -> dict:
    champion_version, champion_metrics = get_champion(client)
    candidate_metric = candidate_metrics[metric_key]

    if champion_version is None:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, candidate_version)
        return {
            "promoted": True,
            "reason": "no_existing_champion",
            "candidate_version": candidate_version,
            "candidate_metric": candidate_metric,
            "champion_version": None,
            "champion_metric": None,
        }

    champion_metric = champion_metrics[metric_key]
    promoted = candidate_metric > champion_metric

    if promoted:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, candidate_version)
        reason = f"candidate {metric_key} {candidate_metric:.4f} > champion {champion_metric:.4f}"
    else:
        client.set_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS, candidate_version)
        reason = f"candidate {metric_key} {candidate_metric:.4f} <= champion {champion_metric:.4f}"

    return {
        "promoted": promoted,
        "reason": reason,
        "candidate_version": candidate_version,
        "candidate_metric": candidate_metric,
        "champion_version": champion_version,
        "champion_metric": champion_metric,
    }
