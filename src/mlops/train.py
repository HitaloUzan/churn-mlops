"""Trains and registers a churn model in MLflow — reuses Project 1's
pipeline (cleaning, feature engineering, XGBoost) so this project's story is
literally "the model already in production, now tracked."

Run with: python -m src.mlops.train
Registers a new version of `churn-classifier` in the MLflow Model Registry
(sqlite:///mlflow.db) and, if no `champion` alias exists yet, promotes it —
the very first run always becomes champion by definition (see
docs/decisions.md ADR-002 on aliases vs. stages).
"""

import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mlops.config import (
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    RANDOM_STATE,
)
from mlops.data_splits import X_y, make_splits
from pipeline.features import build_feature_pipeline, build_preprocessor

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
}


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("business_features", build_feature_pipeline()),
            ("preprocess", build_preprocessor()),
            ("clf", XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)),
        ]
    )


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_and_evaluate(train_df, eval_df) -> tuple[Pipeline, dict]:
    X_train, y_train = X_y(train_df)
    X_eval, y_eval = X_y(eval_df)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_eval, y_eval)
    return pipeline, metrics


def log_and_register(pipeline: Pipeline, metrics: dict, run_name: str, extra_params: dict | None = None):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(XGB_PARAMS)
        if extra_params:
            mlflow.log_params(extra_params)
        mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=None,
            # skops (mlflow's default) refuses to serialize our custom
            # FunctionTransformer callable and XGBoost's Booster without an
            # explicit trust allowlist — see docs/decisions.md ADR-004.
            serialization_format="cloudpickle",
        )
        return run.info.run_id, model_info.registered_model_version


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Building fixed reference/eval_holdout/current_raw splits...")
    splits = make_splits()
    print(f"  reference: {len(splits['reference'])} rows")
    print(f"  eval_holdout: {len(splits['eval_holdout'])} rows (never trained on)")
    print(f"  current_raw: {len(splits['current_raw'])} rows (held for drift simulation)")

    print("Training initial model on 'reference' (simulates the Project 1 model)...")
    t0 = time.time()
    pipeline, metrics = train_and_evaluate(splits["reference"], splits["eval_holdout"])
    train_seconds = round(time.time() - t0, 2)
    print(f"  metrics on eval_holdout: {metrics}")

    run_id, version = log_and_register(
        pipeline, metrics, run_name="initial-deploy", extra_params={"train_seconds": train_seconds}
    )
    print(f"Registered {MODEL_NAME} v{version} (run {run_id})")

    client = MlflowClient()
    try:
        client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        print(f"A '{CHAMPION_ALIAS}' already exists — not auto-promoting. Use src/mlops/pipeline.py.")
    except mlflow.exceptions.MlflowException:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
        print(f"No existing champion — promoted v{version} to '{CHAMPION_ALIAS}'.")


if __name__ == "__main__":
    main()
