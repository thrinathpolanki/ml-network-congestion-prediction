# 🚀 Phase 13: FastAPI Backend

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `api/main.py` | App factory, startup model loading, request-ID, logging and metrics middleware |
| `api/routes.py` | The endpoints |
| `api/schemas.py` | Pydantic request and response models |
| `api/state.py` | Reading buffer, prediction log, metrics registry |
| `api/errors.py` | One structured error shape for every failure |
| `api/security.py` | Optional API-key check |
| `scripts/run_api.py` | Starts the server with the host and port from config |
| `tests/test_api.py` | 18 new tests (184 total, all passing) |

Modified: `configs/config.yaml` (new `api:` section), `src/features/engineering.py`, `tests/conftest.py` and the other changed tests. It also includes a refreshed `models/feature_builder.joblib`, which now records each site's region.

---

## ⚙️ Commands Executed

### Step 1: Run automated tests

```powershell
python -m pytest -q
```

**Result:**

```text
.......................................................................................................................................                 [100%]
135 passed in 88.70s (0:01:28)
```

### Step 2: Install uvicorn and FastAPI

```powershell
python -m pip install uvicorn
python -m pip install fastapi "uvicorn[standard]"
```

**Result:** The uvicorn and fastapi libraries were installed.

### Step 3: Start the API server

```powershell
python scripts/run_api.py  #then http://localhost:8000/docs
```

**Result:** The server opens on localhost.

```text
INFO:     Started server process [26212]
INFO:     Waiting for application startup.
2026-10-18 19:52:43,140 | INFO     | src.alerts.notifiers | Webhook notifier not configured (set ALERT_WEBHOOK_URL).
2026-10-18 19:52:43,140 | INFO     | src.alerts.notifiers | Email notifier not configured (set SMTP_HOST, ALERT_EMAIL_TO, ALERT_EMAIL_FROM).
2026-10-18 19:52:43,140 | INFO     | src.alerts.service | Alert service ready; channels: ['log']
2026-10-18 19:52:43,141 | INFO     | src.prediction.predictor | Serving model from production_model.json: lstm
2026-10-18 19:52:43,178 | INFO     | src.prediction.predictor | Predictor ready: model=lstm, horizon=6 steps, tau=0.160, history=36 steps
2026-10-18 19:52:43,179 | INFO     | api | Restored alert state from Q:\Network-Congestion\ai-network-congestion-prediction\logs\alert_state.json
2026-10-18 19:52:43,180 | INFO     | api | API started: model=lstm, alert channels=['log'], API key protection disabled
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
