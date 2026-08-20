# Architecture Decision Records

## ADR-001: Pipeline logic copied from Project 1, not imported across repos

`src/pipeline/{ingest,clean,features}.py` are the same files as
[churn-pipeline](https://github.com/HitaloUzan/churn-pipeline), copied rather than
imported as a cross-repo dependency. The roadmap requires each portfolio project to be
an independent Git repository ("Cada projeto é um repositório Git independente") — a
cross-repo import would violate that, and for a portfolio-scope project a private
package registry is more infrastructure than the demonstration needs. The trade-off
(a bugfix in Project 1's cleaning logic wouldn't automatically propagate here) is
explicitly accepted, not hidden.

## ADR-002: MLflow aliases, not stages

`set_registered_model_alias` is used throughout (`champion` / `challenger`), not
`transition_model_version_stage`. Both exist in MLflow 3.15.1, but stages are the
older, deprecated mechanism — aliases are what MLflow's own current documentation
recommends. Mapping to the roadmap's vocabulary: `champion` = "production" (the model
actually serving `/predict`), `challenger` = "staging" (a candidate that hasn't beaten
the champion yet).

## ADR-003: Fixed three-way split — eval_holdout / reference / current_raw

Every script in this project (`src/mlops/data_splits.py`) uses the same split, seeded
once: 20% carved out as `eval_holdout` and **never trained on** — the one constant
yardstick every model version is scored against — then the remaining 80% splits into
`reference` (what the "already deployed" model trains on) and `current_raw` (held-out
data simulating what arrives after deploy). `monitoring/simulate_drift.py` optionally
perturbs `current_raw` into `current_drifted`. Without a fixed, shared eval set,
comparing champion vs. candidate metrics wouldn't mean anything — they'd each be
scored on different data.

## ADR-004: cloudpickle serialization, not MLflow's default (skops)

`mlflow.sklearn.log_model` defaults to `skops`, a serialization format designed to be
safer than pickle against arbitrary code execution on load. It rejected this project's
model outright: `add_business_features` (a custom `FunctionTransformer` callable) and
XGBoost's `Booster`/`XGBClassifier` aren't in skops' default trusted-type allowlist,
and it refuses to serialize them without an explicit trust list. `cloudpickle` was
used instead — the trade-off (cloudpickle can execute arbitrary code on
deserialization, which matters for models loaded from an untrusted source) is
acceptable here because this is a single-operator local registry, not a multi-tenant
serving platform loading arbitrary third-party models.

## ADR-005: Importance-weighted drift trigger, not a blanket "% of columns" threshold

Verified empirically, not assumed: a naive "retrain if ≥30% of columns show
significant drift" threshold **failed to catch a real, deliberately injected drift**
in testing. The simulation shifted exactly 3 of 19 feature columns (`Contract`,
`tenure`, `MonthlyCharges`) — precisely Project 1's top SHAP features — but 3/19 ≈
15.8%, under any reasonable blanket threshold, even though those 3 columns are the
ones the model actually depends on most. `src/monitoring/drift.py` now triggers
retraining if **either** the blanket share crosses the threshold **or** any of a
short list of critical features (drawn from Project 1's `shap_top_features.json`)
individually drifts past its own per-column threshold. This is the single most
important finding of this project: a monitoring system that treats every column as
equally important will miss the drift that actually matters.

## ADR-006: Promotion requires strictly better recall — ties don't promote

`src/mlops/governance.py::promote_if_better` promotes only if
`candidate_metric > champion_metric` (strict inequality) on `recall` — the same
priority metric Project 1 optimized for (a missed churner costs more than a false
positive). A tie keeps the existing champion; "no worse" is not the bar, "better" is.
In the one real run recorded in this repo's committed state, the candidate trained on
drift-injected data actually scored *worse* (recall 0.8204 vs. champion 0.8311) and
was correctly rejected — see the README for why that's a feature of the demo, not a
disappointing result: it's proof the governance gate isn't a rubber stamp.

## ADR-007: MLflow registry state (mlflow.db, mlruns/) committed to the repo

Same reasoning as the trained-model artifacts committed in Projects 1–3: cloning this
repo and running `uvicorn` should produce a working `/predict`, `/model-history`, and
`/drift-status` immediately, without first requiring `python -m src.mlops.train`. The
committed state reflects one real run of `src/mlops/train.py` followed by one real run
of `src/mlops/pipeline.py` (with drift injected) — champion v1, rejected challenger
v2 — reproducible by anyone via the fixed random seeds in `mlops/config.py`.

## ADR-008: The "observability dashboard" is MLflow's own UI plus a small API — not a rebuilt frontend

The roadmap asks for a "dashboard de observabilidade" showing model performance
history. Rather than reimplementing a UI MLflow already ships
(`mlflow ui --backend-store-uri sqlite:///mlflow.db`, which browses every run, metric,
and registered version out of the box), this project exposes the same registry state
programmatically via `GET /model-history` and `GET /drift-status` — useful for
integrating into another system, while `mlflow ui` remains the actual dashboard for a
human. Building a custom frontend to duplicate MLflow's own UI would be effort spent
re-deriving a tool that already exists, which project 5 is a much better showcase for.

## ADR-009: Docker trains fresh inside the container — doesn't copy the host's mlflow.db

Verified before shipping, not assumed safe: MLflow's SQLite backend stores each run's
`artifact_uri` as an **absolute path resolved at the moment the run was logged** —
inspecting the committed `mlflow.db` shows entries like
`file:C:/Users/Hitalo Uzan/OneDrive/.../churn-mlops/mlruns/1/<run_id>/artifacts`. That
path is meaningless inside a Linux container. Copying the host-generated `mlflow.db`
and `mlruns/` into the image verbatim would build successfully and then fail at
`/predict` time with a "model file not found" error — a bug that a build-succeeded
CI check would not catch. Instead, `Dockerfile`'s `trainer` stage runs
`src.mlops.train` and `src.mlops.pipeline` **inside the container**, so every stored
path is `/app/mlruns/...` and self-consistent. The final image copies only the
resulting `mlflow.db`/`mlruns/`/`pipeline_run_summary.json` out of that stage, not the
heavier training dependencies (evidently, statsmodels) needed to produce them.

## ADR-010: Python 3.14 as the dev environment

Same situation as the other four projects: only Python 3.14.4 is installed here, and
every dependency (mlflow, evidently, xgboost, scikit-learn, fastapi) resolved
prebuilt `cp314` wheels with no compilation issues. `Dockerfile` uses the matching
`python:3.14-slim` base image.
