from unittest.mock import MagicMock

import mlflow

from mlops.governance import get_champion, promote_if_better


def _mock_model_version(version: str, run_id: str):
    mv = MagicMock()
    mv.version = version
    mv.run_id = run_id
    return mv


def _mock_run(metrics: dict):
    run = MagicMock()
    run.data.metrics = metrics
    return run


def test_get_champion_returns_none_when_no_alias_set():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("not found")

    version, metrics = get_champion(client)

    assert version is None
    assert metrics is None


def test_get_champion_returns_version_and_metrics():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _mock_model_version("3", "run-3")
    client.get_run.return_value = _mock_run({"recall": 0.9})

    version, metrics = get_champion(client)

    assert version == "3"
    assert metrics == {"recall": 0.9}


def test_promote_if_better_promotes_when_no_champion_exists():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("not found")

    result = promote_if_better(client, candidate_version="1", candidate_metrics={"recall": 0.7})

    assert result["promoted"] is True
    assert result["reason"] == "no_existing_champion"
    client.set_registered_model_alias.assert_called_once_with("churn-classifier", "champion", "1")


def test_promote_if_better_promotes_when_candidate_beats_champion():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _mock_model_version("1", "run-1")
    client.get_run.return_value = _mock_run({"recall": 0.70})

    result = promote_if_better(client, candidate_version="2", candidate_metrics={"recall": 0.85})

    assert result["promoted"] is True
    client.set_registered_model_alias.assert_called_once_with("churn-classifier", "champion", "2")


def test_promote_if_better_rejects_when_candidate_is_worse():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _mock_model_version("1", "run-1")
    client.get_run.return_value = _mock_run({"recall": 0.85})

    result = promote_if_better(client, candidate_version="2", candidate_metrics={"recall": 0.70})

    assert result["promoted"] is False
    client.set_registered_model_alias.assert_called_once_with("churn-classifier", "challenger", "2")


def test_promote_if_better_rejects_when_candidate_ties_champion():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _mock_model_version("1", "run-1")
    client.get_run.return_value = _mock_run({"recall": 0.80})

    result = promote_if_better(client, candidate_version="2", candidate_metrics={"recall": 0.80})

    assert result["promoted"] is False
