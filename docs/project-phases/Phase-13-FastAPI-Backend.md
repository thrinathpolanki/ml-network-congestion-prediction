# 🚀 Phase 13: FastAPI Backend

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `src/api/main.py` | App factory, startup model loading, request-ID and logging middleware, all endpoints, error handlers |
| `src/api/schemas.py` | Pydantic request and response models (`Reading` mirrors the 14 KPIs of the data contract) |
| `src/api/state.py` | Thread-safe shared state: predictor, alert service, latest result per site, counters |
| `tests/test_api.py` | 17 tests (real model, sample data, no network) |

## 🔌 Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness, whether the model is loaded, uptime |
| `GET /model/info` | Model name, horizon, required history, risk thresholds, alert channels |
| `GET /model/performance` | Held-out test table and selection record from the report file |
| `POST /predict` | Readings of one site in, prediction and any alerts out |
| `POST /predict/batch` | Many sites; sites with too little history are listed in `skipped_sites`, never fail the request |
| `POST /predict/history` | Backtest for one site (a prediction at every timestamp with enough history), never raises alerts |
| `GET /alerts` | Filter by site, minimum severity, kind; newest first |
| `POST /alerts/{id}/acknowledge` | Mark an alert acknowledged (404 if unknown) |
| `GET /sites` | Sites seen since startup with latest assessment and whether an alert is active |
| `GET /metrics` | Request, prediction, error and alert counters, average scoring latency, suppression counts |

---

## ⚙️ Commands Executed

### Step 1: Start the API server

```powershell
uvicorn src.api.main:app --port 8000
```

**Result:**

```text
INFO:     Started server process [5452]
INFO:     Waiting for application startup.
2026-10-24 22:53:00,953 | INFO     | src.api | Starting API: loading model artifacts
2026-10-24 22:53:00,973 | INFO     | src.prediction.predictor | Serving model from production_model.json: lstm
2026-10-24 22:53:01,517 | INFO     | src.prediction.predictor | Predictor ready: model=lstm, horizon=6 steps, tau=0.160, history=36 steps
2026-10-24 22:53:01,519 | INFO     | src.alerts.notifiers | Webhook notifier not configured (set ALERT_WEBHOOK_URL).
2026-10-24 22:53:01,519 | INFO     | src.alerts.notifiers | Email notifier not configured (set SMTP_HOST, ALERT_EMAIL_TO, ALERT_EMAIL_FROM).
2026-10-24 22:53:01,520 | INFO     | src.alerts.service | Alert service ready; channels: ['log']
2026-10-24 22:53:01,537 | INFO     | src.api | Restored alert engine state from Q:\Network-Congestion\ai-network-congestion-prediction\logs\alert_state.json
2026-10-24 22:53:01,537 | INFO     | src.api | API ready: model=lstm
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Step 2: Check the health endpoint

```powershell
curl.exe http://localhost:8000/health       #open in another powershell without quitting the UVICORN powershell
```

**Result:**

```json
{"status":"ok","model_loaded":true,"model":"lstm","uptime_seconds":17.9}
```

### Step 3: Install httpx2

```powershell
python -m pip install httpx2
```

**Result:** The httpx2 library was installed.

### Step 4: Run the API tests

```powershell
python -m pytest tests/test_api.py -q
```

**Result:**

```text
...............                                                                                                                      [100%]
15 passed in 6.28s
```
