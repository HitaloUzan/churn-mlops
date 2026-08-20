import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["champion_loaded"] is True


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in ("Yes", "No")
    assert body["model_alias"] == "champion"


def test_predict_rejects_invalid_tenure(client):
    bad_payload = {**VALID_PAYLOAD, "tenure": -1}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_model_history_lists_champion_and_challenger(client):
    response = client.get("/model-history")
    assert response.status_code == 200
    body = response.json()
    aliases = {v["alias"] for v in body}
    assert "champion" in aliases


def test_model_history_ordered_newest_first(client):
    response = client.get("/model-history")
    versions = [int(v["version"]) for v in response.json()]
    assert versions == sorted(versions, reverse=True)


def test_drift_status_returns_last_pipeline_run(client):
    response = client.get("/drift-status")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["drift_detected"], bool)
