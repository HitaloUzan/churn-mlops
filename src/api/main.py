"""Observability API — serves predictions from whichever model version
currently holds the `champion` alias in the MLflow Model Registry, plus
version history and the last drift-check result. This is the "dashboard"
requested by the roadmap: rather than reimplementing what MLflow's own UI
(`mlflow ui --backend-store-uri sqlite:///mlflow.db`) already provides,
this API exposes the same registry state programmatically (see
docs/decisions.md ADR-008).
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow.tracking import MlflowClient

from api.schemas import (
    CustomerFeatures,
    DriftStatusResponse,
    HealthResponse,
    ModelVersionInfo,
    PredictionResponse,
)
from mlops.config import CHAMPION_ALIAS, MLFLOW_TRACKING_URI, MODEL_NAME

ROOT = Path(__file__).resolve().parents[2]
RUN_SUMMARY_PATH = ROOT / "models" / "pipeline_run_summary.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("churn-mlops-api")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    state["client"] = MlflowClient()
    state["model"] = None
    state["champion_version"] = None

    try:
        version = state["client"].get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
        state["champion_version"] = str(version.version)
        logger.info(f"Loaded champion: {MODEL_NAME} v{version.version}")
    except mlflow.exceptions.MlflowException:
        logger.warning(f"No '{CHAMPION_ALIAS}' model found — run src/mlops/train.py first")

    yield
    state.clear()


app = FastAPI(
    title="Churn MLOps Observability API",
    description="Serves the champion model and exposes MLflow registry + drift monitoring state.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        json.dumps(
            {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
        )
    )
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        champion_loaded=state.get("model") is not None,
        champion_version=state.get("champion_version"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if state.get("model") is None:
        raise HTTPException(status_code=503, detail="No champion model loaded — run src/mlops/train.py first")

    X = pd.DataFrame([customer.model_dump()])
    probability = float(state["model"].predict_proba(X)[0, 1])

    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction="Yes" if probability >= 0.5 else "No",
        model_version=state["champion_version"],
    )


@app.get("/model-history", response_model=list[ModelVersionInfo])
def model_history():
    client = state["client"]
    try:
        registered_model = client.get_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        raise HTTPException(status_code=503, detail=f"Model '{MODEL_NAME}' not registered yet")

    version_to_alias = {str(v): alias for alias, v in registered_model.aliases.items()}
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    history = []
    for v in sorted(versions, key=lambda v: int(v.version), reverse=True):
        run = client.get_run(v.run_id)
        history.append(
            ModelVersionInfo(
                version=str(v.version),
                alias=version_to_alias.get(str(v.version)),
                run_name=run.data.tags.get("mlflow.runName"),
                metrics=dict(run.data.metrics),
                created_at=datetime.fromtimestamp(v.creation_timestamp / 1000, tz=timezone.utc).isoformat(),
            )
        )
    return history


@app.get("/drift-status", response_model=DriftStatusResponse)
def drift_status():
    if not RUN_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=503, detail="No pipeline run yet — run src/mlops/pipeline.py first"
        )
    with open(RUN_SUMMARY_PATH) as f:
        summary = json.load(f)
    return DriftStatusResponse(**summary)
