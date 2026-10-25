# 🚀 Phase 15: System Integration

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `tests/test_integration.py` | 8 tests tying the chain together and guarding the architecture |
| `scripts/smoke_test.py` | End-to-end check against a running API; exit code 0 means every step passed |

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

### Step 2: Run the smoke test

```powershell
python scripts/smoke_test.py
```

**Result:**

```text
Smoke test against http://localhost:8000
  [PASS] health: model loaded  (lstm)
  [PASS] model/info: horizon and history reported  (30 min, 36 steps)
  [PASS] predict: valid probability and risk level  (p=0.0004)
  [PASS] predict/batch: every site scored  (20 sites)
  [PASS] sites: registry populated
  [PASS] metrics: counters moving
  [PASS] alerts: endpoint reachable  (0 raised by this batch)
  [PASS] invalid input: rejected with 422  (validation_error)

8/8 checks passed
```
