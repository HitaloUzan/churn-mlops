"""Pydantic request/response schemas for the observability API."""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Same input shape as Project 1's churn-pipeline API."""

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: str
    model_version: str
    model_alias: str = "champion"


class ModelVersionInfo(BaseModel):
    version: str
    alias: str | None
    run_name: str | None
    metrics: dict
    created_at: str


class DriftStatusResponse(BaseModel):
    timestamp_utc: str | None = None
    drift_mode: str | None = None
    drift_detected: bool | None = None
    trigger_reason: str | None = None
    drift_share: float | None = None
    drifted_critical_features: list[str] = []
    retrained: bool | None = None
    promotion: dict | None = None


class HealthResponse(BaseModel):
    status: str
    champion_loaded: bool
    champion_version: str | None
