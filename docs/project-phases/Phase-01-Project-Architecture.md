# 🚀 Phase 1: Project Architecture

**Project:** Network Congestion Prediction & Alert System

---

## 📑 Table of Contents

- [1. Key Design Decisions](#1-key-design-decisions)
- [2. Data Flow](#2-data-flow)
- [3. Component Responsibilities](#3-component-responsibilities)
- [4. Recommended Directory Structure](#4-recommended-directory-structure)

---

## 1. Key Design Decisions

These decisions shape everything that follows, so I fixed them at the start of the project.

| Decision | Choice | Why |
| --- | --- | --- |
| Sampling interval | 5 minutes (configurable) | Fine enough to see peaks and spikes, small enough for a laptop. |
| Prediction horizon | 30 minutes = 6 steps (configurable) | The model predicts the state at t + H using only data up to t. |
| Target | 4-class multiclass: Normal / Warning / Congested / Critical | One model gives both severity and probability. |
| Congestion probability | P(Congested) + P(Critical) from the class probabilities | Gives a single risk number without a second model. |
| Label source | Weighted congestion score from latency, packet loss, utilization, throughput drop and so on, with configurable thresholds | Principled and tunable, not an arbitrary cutoff. |
| Label leakage control | The score is computed from future metrics at t+H and is never used as a feature | Prevents the model from just re-deriving its own label. |
| Splitting | Chronological per site, e.g. 70/15/15, with a gap of H steps between splits | Prevents temporal leakage. |
| Preprocessing and features | One shared module used by training and inference | Avoids train/serve skew. |
| Model selection | Decided by measured results, not assumed | XGBoost may well beat the LSTM on tabular lag features. |
| Dashboard | Streamlit, calling the FastAPI service | Simple and professional, and keeps the API as the single source of truth. |
| Alert delivery | Log + API + dashboard first, webhook/email optional via env vars | Manageable first version. |

> 📝 **Design point:** For real-time inference, the API needs recent history per site (lags and rolling windows). Two modes are supported:
>
> - `/predict` takes a window of recent readings for one site, which is stateless and easy to test.
> - An in-memory per-site ring buffer lets the dashboard or a simulator stream one reading at a time.

---

## 2. Data Flow

```text
 ┌────────────────────────┐
 │ DATA SOURCE (swappable)│  SyntheticSource (now) | CSVSource | future: Kafka/Prometheus/OSS
 └──────────┬─────────────┘
            │  standard schema: timestamp, site_id, KPIs...
            ▼
 ┌────────────────────────┐
 │ INGESTION + VALIDATION │  schema check, dtype coercion, range checks
 └──────────┬─────────────┘
            ▼
 ┌──────────────────────┐
 │ PREPROCESSING        │  dedupe, sort, missing values, outliers, resample, scale
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ FEATURE ENGINEERING  │  lags, rolling stats, EWM, time encodings, prev state
 └──────────┬───────────┘
            ▼
 ┌─────────────────────────────────────────────┐
 │ MODELS: Baselines | XGBoost | LSTM          │
 └──────────┬──────────────────────────────────┘
            ▼
 ┌──────────────────────┐
 │ RISK ENGINE          │  class probs → congestion probability → risk level + action
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ ALERT ENGINE         │  thresholds, cooldown, dedupe, escalation, recovery
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ FASTAPI              │  /predict /alerts /sites /metrics ...
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ STREAMLIT DASHBOARD  │  operator view
 └──────────────────────┘
```

---

## 3. Component Responsibilities

| Component | Module | Responsibility |
| --- | --- | --- |
| Data source | `src/data/sources.py`, `src/data/generator.py` | Abstract `DataSource` interface. Synthetic and CSV implementations. This is the interface to real telecom data later. |
| Ingestion | `src/data/schema.py` | Pydantic/pandas schema and validation of incoming metrics. |
| Preprocessing | `src/preprocessing/` | Cleaning, imputation, outlier capping, chronological split, fitted scaler. |
| Labeling | `src/features/labeling.py` | Congestion score → 4 states, configurable thresholds. |
| Features | `src/features/engineering.py` | Single `build_features()` used by training and API. |
| Baselines | `src/models/baselines.py` | Majority class, threshold detector, persistence (naive temporal). |
| XGBoost | `src/models/xgb_model.py` | Training, tuning, class weights, importance. |
| LSTM | `src/models/lstm_model.py` | Sequence builder, PyTorch LSTM, early stopping, checkpoints. |
| Evaluation | `src/models/evaluate.py` | Metrics, confusion matrices, timing, comparison table. |
| Prediction | `src/prediction/predictor.py` | Loads artifacts, runs the full inference pipeline. |
| Risk engine | `src/prediction/risk.py` | Probabilities → risk level and recommended action. |
| Alerts | `src/alerts/engine.py`, `notifiers.py` | Alert logic and delivery channels. |
| API | `api/` | FastAPI app, schemas, routes, error handlers. |
| Dashboard | `dashboard/app.py` | Streamlit UI. |
| Utils | `src/utils/` | Config loader, logging, seeding. |

> **LSTM framework:** The project specification lists no deep learning library, so I chose PyTorch (CPU). It is easy to read and runs fine on a laptop. TensorFlow/Keras is a reasonable alternative if the institution prefers it.

---

## 4. Recommended Directory Structure

I made three changes to the original template:

- Added `src/config.py` plus `configs/config.yaml` for central configuration.
- Added a `src/pipeline/` folder for runnable training steps.
- Added `artifacts/` conventions inside `models/` for the scaler, feature list, and metrics.

```text
ai-network-congestion-prediction/
│
├── configs/
│   └── config.yaml              # all thresholds, horizon, paths, model params
│
├── data/
│   ├── raw/                     # generated or real raw data (gitignored)
│   ├── processed/               # cleaned + featured data (gitignored)
│   └── sample/                  # small committed sample for tests/demo
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── config.py                # loads YAML + env overrides
│   ├── data/                    # sources.py, generator.py, schema.py
│   ├── preprocessing/           # cleaning.py, splitting.py, scaling.py
│   ├── features/                # labeling.py, engineering.py
│   ├── models/                  # baselines.py, xgb_model.py, lstm_model.py, evaluate.py
│   ├── prediction/              # predictor.py, risk.py, history_store.py
│   ├── alerts/                  # engine.py, notifiers.py
│   ├── pipeline/                # train.py, compare.py (orchestration)
│   └── utils/                   # logging_config.py, seed.py
│
├── api/
│   ├── main.py                  # app factory, lifespan, startup model load
│   ├── schemas.py               # Pydantic models
│   ├── routes.py
│   └── errors.py
│
├── dashboard/
│   └── app.py
│
├── tests/
│   ├── test_generator.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_risk_alerts.py
│   └── test_api.py
│
├── models/                      # saved model files + scaler + metadata (gitignored)
├── reports/                     # figures, metrics.json, comparison table
├── logs/
├── scripts/
│   ├── generate_data.py
│   ├── run_eda.py
│   ├── train_all.py
│   └── demo.py
│
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml           # api + dashboard services
├── README.md
├── docs/ACADEMIC_REPORT.md
└── main.py                      # CLI entry: generate | train | serve | demo
```
