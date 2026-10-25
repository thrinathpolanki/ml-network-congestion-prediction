# 🚀 Phase 14: Dashboard

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `dashboard/app.py` | Streamlit app with five tabs |
| `src/dashboard/client.py` | API client and replay helpers (no Streamlit imports, so unit-testable) |
| `tests/test_dashboard_client.py` | 4 tests for the helpers |
| `src/api/main.py` | Gained `POST /predict/history` and `GET /model/performance` for the dashboard |

## 🖥️ Screens

| Tab | Shows |
| --- | --- |
| Network overview | Sites monitored, at risk, CRITICAL, active alerts, average scoring time; risk bar chart; site table; map |
| Site detail | Utilization, latency, packet loss, traffic volume trends; prediction history for the chosen site |
| Alerts | Filter by severity and kind; suppression counts |
| Model performance | Held-out test table with a metric chooser and a simulated-data warning |
| How it works | Pipeline, risk thresholds and scope, written for an evaluator |

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

### Step 2: Start the Streamlit dashboard

```powershell
streamlit run dashboard/app.py
```

**Result:**

```text
2026-10-24 23:09:42.507 Uvicorn server started on :::8501

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.60.18.41:8501
```
