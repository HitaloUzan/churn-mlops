# --- trainer: trains fresh INSIDE the container so MLflow's stored artifact
# paths are container-native (/app/mlruns/...), not the host's absolute
# Windows path baked into a locally-generated mlflow.db — see
# docs/decisions.md ADR-009 for why copying the host's mlflow.db/mlruns
# in verbatim would silently break model loading at runtime.
FROM python:3.14-slim AS trainer

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY src/ src/
COPY data/raw/ data/raw/
RUN python -m src.mlops.train && python -m src.mlops.pipeline

# --- runtime-deps: install only what the API needs to serve predictions ---
FROM python:3.14-slim AS runtime-deps

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- final: slim runtime image, no compilers/build tools, no training deps ---
FROM python:3.14-slim

WORKDIR /app

COPY --from=runtime-deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ src/
COPY --from=trainer /app/mlflow.db mlflow.db
COPY --from=trainer /app/mlruns/ mlruns/
COPY --from=trainer /app/models/pipeline_run_summary.json models/pipeline_run_summary.json

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
ENV PYTHONPATH=/app/src

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
