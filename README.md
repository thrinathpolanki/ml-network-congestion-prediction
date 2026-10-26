# AI-Based Network Congestion Prediction and Alert System

Predicts, about **30 minutes ahead**, whether a mobile-network cell will become congested, turns the prediction into a risk level, and raises (de-duplicated) alerts, with a REST API and a dashboard on top.

> **Everything here is trained and evaluated on SIMULATED data.** No real telecom measurements were used. The results show that the method works on data with these patterns. They are **not** evidence of performance on a real network.

---

## 1. Problem statement
Mobile cells become congested at busy times: latency and packet loss rise and throughput per user falls. Operators usually find out after users are affected. The task is to forecast, from a cell's recent telemetry, whether it will be congested 30 minutes from now, early enough to act.

## 2. Motivation
Reacting to congestion is expensive and late. A reliable early warning lets an operator rebalance load or adjust QoS beforehand. Two practical difficulties shape the design:
- congestion is **rare** (most of the time cells are healthy), so plain accuracy is misleading;
- naive alerting is **noisy**: a probability that flickers around a threshold produces dozens of notifications per cell per day.

## 3. Objectives
1. Generate realistic, clearly labelled synthetic telemetry (the real data is not available).
2. Build a leak-free pipeline: cleaning, features, chronological train/validation/test split.
3. Compare baselines, XGBoost, an LSTM and a hybrid fairly, and choose the production model **before** looking at test results.
4. Convert probabilities to risk levels and alerts with debounce, cooldown and recovery.
5. Serve it through an API and a dashboard, with tests, logging and Docker.

## 4. Proposed solution
A pipeline from raw KPI readings to alerts:

1. Clean each cell's readings and put them on a 5-minute grid.
2. Build features from the recent past (lags, rates of change, rolling statistics, calendar).
3. A model outputs the probability that the cell is **Congested or Critical at t + 30 min**.
4. The probability maps to a risk level (LOW, MEDIUM, HIGH, CRITICAL) with a recommended action.
5. The alert engine decides whether to notify (log, webhook, email).

## 5. System architecture

```
 telemetry (CSV / API)                                             ┌───────────────┐
        │                                                          │  Dashboard    │
        ▼                                                          │  (Streamlit)  │
 ┌──────────────┐   ┌────────────┐   ┌──────────┐   ┌───────────┐  └──────▲────────┘
 │ FastAPI      │──▶│ cleaning + │──▶│ model    │──▶│ risk      │         │ HTTP
 │ validation   │   │ features   │   │ (hybrid) │   │ levels    │         │
 └──────┬───────┘   └────────────┘   └──────────┘   └─────┬─────┘         │
        │                                                  ▼               │
        │                                           ┌─────────────┐        │
        └──────────────────────────────────────────▶│ alert engine│────────┘
                                                    │ + notifiers │  log / webhook / email
                                                    └─────────────┘
```
- `src/data`, `src/preprocessing`, `src/features`: data, cleaning, features (shared by training and serving).
- `src/models`: baselines, XGBoost, LSTM, hybrid, evaluation and comparison.
- `src/prediction`: the predictor (raw readings to a risk assessment) and the risk engine.
- `src/alerts`: engine, store, notifiers, service.
- `src/api`: FastAPI app. `dashboard/` and `src/dashboard`: Streamlit app and API client.

The API and dashboard contain **no ML or preprocessing code** of their own (a test enforces this), so training and serving cannot drift apart.

## 6. Technologies
Python 3.12, pandas, NumPy, SciPy, scikit-learn, XGBoost, PyTorch (CPU), FastAPI, Uvicorn, Streamlit, Altair, pytest, Docker and Docker Compose.

## 7. Dataset description
Simulated telemetry for **20 cells** at a **5-minute** interval from 2026-01-01 to 2026-03-02 (about 340,000 rows), in five regions: urban core, urban residential, business district, suburban, rural. Each row has 14 KPIs:

`active_users`, `active_sessions`, `traffic_volume_gb`, `bandwidth_utilization_pct`, `download_throughput_mbps`, `upload_throughput_mbps`, `latency_ms`, `packet_loss_pct`, `jitter_ms`, `signal_quality_sinr_db`, `cpu_utilization_pct`, `dropped_call_rate_pct`, `handover_failure_rate_pct`, `retransmission_rate_pct`

plus identifiers (timestamp, site id), static attributes (region, capacity, latitude, longitude) and a `data_source` column that says `synthetic`. A small sample (2 days, 20 cells) is committed in `data/sample/`.

**Congestion label.** Each KPI is mapped to a 0–1 sub-score between a "good" and a "bad" value (for example latency 45 ms to 200 ms). A weighted mean, blended with the worst single sub-score (30%), gives a congestion score. Score boundaries define four states: Healthy, Warning (≥ 0.25), Congested (≥ 0.45), Critical (≥ 0.70). The prediction target is the state **30 minutes later**; the binary "congested" target is Congested or Critical. All weights and cut-offs are in `configs/config.yaml`.

## 8. Synthetic data methodology
For each cell: offered load = peak load × daily profile × day-to-day level × autocorrelated noise × events. The daily profile depends on the cell type (residential evening peak, business daytime peaks, weekday vs weekend). Utilization is a soft-saturating function of load, and latency, loss, jitter, drops and retransmissions grow non-linearly as utilization approaches saturation while throughput per user falls. The generator also adds short surges (10–45 min), sustained high-load episodes (2–6 h), and deliberate **data-quality problems** (glitches, missing values, duplicate rows, impossible values, dropped timestamps) so the cleaning step has real work to do.

Because the labels are computed from the same KPIs the generator produces, the problem is easier and cleaner than real life. That is the most important caveat of this project.

## 9. Feature engineering
Features use only the past (`docs/FEATURES.md` has the full list):
- **Lags** of key KPIs: 5, 10, 15, 30, 60 minutes.
- **Rates of change** over 1, 3 and 6 steps.
- **Rolling mean, std, max, min** over 30 min, 1 h and 3 h, and **exponentially weighted** means.
- **Previous congestion state/score** and the share of the last 1 h / 3 h spent congested.
- **Calendar**: hour, day of week, weekend, peak hour.
- **Static**: region and capacity. Cell position was tested and **left out** (`docs/SPATIAL_FEATURES.md`): validation gain was about 0.001 PR-AUC and it may only help the model recognise training cells.

Cleaning is **causal** online: range checks, gaps of up to 30 minutes filled forward, longer gaps left unscorable.

## 10. Model methodology
- The data is split **chronologically**: first 70% train, next 15% validation, last 15% test.
- Scaling and class weights are fitted on training data only.
- Each model outputs probabilities for the four states; the **congestion probability** is P(Congested) + P(Critical).
- Probabilities are **calibrated** (isotonic regression) on the validation period, and the alert threshold τ is chosen on validation by maximizing F2 (recall counts twice as much as precision).
- The hybrid averages the XGBoost and LSTM probability vectors and has its own calibration and threshold.

## 11. XGBoost
Gradient-boosted decision trees on the engineered tabular features. Each tree corrects the errors of the previous ones; the model is strong on tabular data and fast. Here: a four-class objective, up to 600 trees with early stopping on validation, square-root class weights to offset the rarity of congestion, and a small random search (8 trials, validation PR-AUC) over depth, learning rate, regularization and sampling.

## 12. LSTM
A recurrent neural network (PyTorch, CPU) that reads the **last 3 hours** (36 steps of 5 minutes) of KPI history as a sequence and keeps a memory of how the load has been evolving. One LSTM layer with 64 hidden units, dropout 0.2, Adam with a reduce-on-plateau learning rate, gradient clipping, early stopping on validation loss, and the same class weighting as XGBoost.

## 13. Evaluation methodology
- Four training seeds (42–45) per model. The dataset is a single simulated one, so seeds vary model initialization and sampling only.
- **Selection rule, fixed in advance and applied on validation only:** keep XGBoost unless a challenger beats it by at least 2 pooled seed-SDs on PR-AUC or on "operator area" (event recall averaged over 0.2–1.0 false alerts per site-day), without being clearly worse on the other.
- The decision is written to `models/production_model.json` **before** the test set is scored, and the test set is scored **once** (a lock file prevents re-scoring by accident).
- Metrics: accuracy, macro-F1 (accuracy is misleading because of the class imbalance), congestion precision/recall/F1, PR-AUC, ROC-AUC, **event recall** (an event is at least 3 consecutive congested target steps, detected if flagged early) and **false alerts per site-day**.
- Baselines: majority class, a bandwidth threshold rule, persistence (state now), and seasonal naive (same time yesterday / last week).

## 14. Results
Held-out **test** set, mean over 4 seeds (baselines are deterministic). "Tuned τ" uses the validation-chosen alert threshold; "argmax" uses the plain most-likely class.

| Model | Macro-F1 | PR-AUC | Cong. recall | Cong. precision | Event recall | False alerts / site-day |
|---|---|---|---|---|---|---|
| Majority class | 0.234 | 0.040 | 0.000 | 0.000 | 0.000 | 0.00 |
| Threshold rule (bandwidth) | 0.620 | 0.698 | 0.711 | 0.629 | 0.493 | 0.74 |
| Persistence (state now) | 0.614 | 0.700 | 0.659 | 0.660 | 0.390 | 0.48 |
| Seasonal naive (1 day) | 0.442 | 0.319 | 0.436 | 0.357 | 0.459 | 0.98 |
| XGBoost (argmax) | 0.673 | 0.791 | 0.665 | 0.758 | 0.474 | 0.38 |
| XGBoost (tuned τ) | 0.623 | 0.791 | 0.935 | 0.526 | 0.894 | 0.91 |
| LSTM (argmax) | 0.674 | 0.788 | 0.781 | 0.702 | 0.656 | 0.30 |
| LSTM (tuned τ) | 0.632 | 0.788 | 0.927 | 0.537 | 0.887 | 0.63 |
| **Hybrid (argmax)** | **0.682** | **0.795** | 0.729 | 0.738 | 0.570 | 0.30 |
| **Hybrid (tuned τ), production** | 0.639 | 0.795 | 0.927 | 0.546 | 0.885 | 0.65 |

> **Which model is in production depends on your trained files.** These numbers come from the reports in the repository when this README was written, where the validation rule selected the **hybrid**. A run on another machine selected the **LSTM** (the API log showed `model=lstm`, 36 steps of history). If you retrain, the selection can change. `docs/MODEL_COMPARISON.md`, regenerated by `scripts/make_comparison_report.py`, is the authoritative table for your artifacts.

**Reading it honestly.**
- The learned models beat every baseline on PR-AUC (about 0.79 against 0.70 for the best baseline).
- The hybrid was **selected on validation** (PR-AUC 0.8193 vs 0.8167 for XGBoost; operator area 0.652 vs 0.559). The test differences between the three learned models are small.
- There is a clear trade-off: with the recall-oriented threshold the models catch about 89% of congestion events but with roughly 0.65 false alerts per site-day.
- Full tables, per-region results and the selection evidence are in `docs/MODEL_COMPARISON.md`.

**Effect of the alert engine** (test period, from `scripts/simulate_alerts.py`): about 3.6 notifications per site-day, against about 21 if every risky prediction were notified. About 90% of congestion events are covered, with a median lead time of about 75 minutes, and false-alert episodes drop from 0.64 to 0.26 per site-day.

## 15. API documentation
Start the server, then open **http://localhost:8000/docs** for the interactive reference.

```bash
uvicorn src.api.main:app --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness and whether the model is loaded |
| `GET /model/info` | Model, horizon, required history (depends on the model: 36 steps = 3 h for the LSTM, 73 steps = 6 h for the hybrid), thresholds, alert channels |
| `GET /model/performance` | Held-out test table and selection record |
| `POST /predict` | Readings of **one** site in, prediction and any alerts out |
| `POST /predict/batch` | Many sites; sites with too little history are listed in `skipped_sites` |
| `POST /predict/history` | Backtest for one site (never raises alerts) |
| `GET /alerts` | Filter by `site_id`, `min_severity`, `kind`, `limit` |
| `POST /alerts/{id}/acknowledge` | Acknowledge an alert |
| `GET /sites` | Sites seen since startup with latest assessment |
| `GET /metrics` | Request, prediction, error and alert counters |

Example (readings abbreviated; the real request needs at least the number of consecutive 5-minute readings reported by `/model/info`):
```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "readings": [{"timestamp": "2026-02-24T08:00:00", "site_id": "CELL_012", "region": "urban_core",
                "site_capacity_mbps": 1000, "bandwidth_utilization_pct": 71.2, "latency_ms": 38.5, "...": "..."}],
  "raise_alerts": true }'
```
Response (shape): `{"prediction": {"site_id", "prediction_time", "target_time", "congestion_probability", "risk_level", "predicted_state", "confidence", "recommended_action", ...}, "alerts": [...], "data_note": "..."}`

Errors share one shape, `{"error", "detail", "request_id"}`: **422** for invalid input or too little history, **503** if the model is not loaded or mismatched, **500** with a generic message for unexpected failures (details go to the log with the request ID). Impossible or missing KPI values are cleaned, not rejected. The API has **no authentication**.

## 16. Dashboard
```bash
streamlit run dashboard/app.py          # http://localhost:8501  (API must be running)
```
Tabs: **Network overview** (counts, risk per site, status table, map), **Site detail** (KPI trends and prediction history), **Alerts** (filterable), **Model performance**, **How it works**. The sidebar replays the sample feed into the API one 5-minute step at a time, so you can watch risk rise through the morning peak and alerts appear under the real debounce and cooldown rules. Environment variables: `API_URL`, `DASHBOARD_FEED` (a CSV to replay).

## 17. Alert mechanism
Risk level comes from the calibrated probability: MEDIUM at or above the model's validated threshold τ (0.14 for the hybrid in the shipped reports, 0.16 for the LSTM; see `/model/info`), HIGH at 0.40, CRITICAL at 0.70. Severity mapping: MEDIUM = WARNING, HIGH = HIGH, CRITICAL = CRITICAL; LOW raises nothing.

| Rule | Setting | Purpose |
|---|---|---|
| Debounce | 2 consecutive risky predictions | One noisy sample does not alert |
| CRITICAL bypass | on | Very likely congestion alerts immediately |
| Cooldown / reminders | 60 min | An unchanged situation is not repeated every 5 minutes |
| Escalation | notified when severity rises | Suppressed only if that same severity was already notified within the cooldown (prevents flapping) |
| Recovery | 3 consecutive LOW | Sends a single "recovered" message |
| Re-arm | 15 min | A WARNING cannot re-fire right after recovery |
| Duplicate input | ignored | Re-sending the same timestamp does nothing |

Channels: **log** (always), **webhook** (severity HIGH or above) and **email** (CRITICAL), the last two only when their environment variables are set (see `.env.example`). Alerts are kept in memory, appended to `logs/alerts.jsonl`, and engine state is saved on shutdown and restored on startup.

## 18. Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # runtime
pip install -r requirements-dev.txt        # + pytest, pytest-cov, httpx (for tests)
cp .env.example .env                       # optional: alert webhook / SMTP settings
```
The trained model files are in `models/` and the test report in `reports/`, so the API and dashboard run immediately.

## 19. Configuration
Everything is in `configs/config.yaml` (generator, congestion definition, cleaning, features, models, risk levels, alert rules, logging). Any value can be overridden with an environment variable `NCP_<SECTION>__<KEY>`, for example:
```bash
NCP_LOGGING__LEVEL=DEBUG uvicorn src.api.main:app --port 8000
NCP_PREDICTION__HORIZON_MINUTES=60 python scripts/build_features.py   # changing the horizon requires rebuilding and retraining
```
**Secrets** (webhook URL and token, SMTP settings) come only from environment variables or `.env`. They are never stored in the config or the image and are masked in logs. The serving model is chosen by `models/production_model.json` (`serving.model: auto`).

## 20. Running the project
**Use the trained model (fastest):**
```bash
uvicorn src.api.main:app --port 8000
streamlit run dashboard/app.py
python scripts/smoke_test.py               # end-to-end check against the running API
```
**Rebuild everything from scratch** (several minutes; the LSTM trains on CPU):
```bash
python scripts/generate_data.py            # simulated telemetry
python scripts/prepare_data.py             # clean + chronological split
python scripts/run_eda.py                  # exploratory analysis (training period only)
python scripts/build_features.py
python scripts/run_baselines.py
python scripts/train_xgboost.py
python scripts/train_lstm.py
python scripts/fit_hybrid.py
python scripts/compare_models.py           # multi-seed comparison; scores the TEST set once
python scripts/make_comparison_report.py   # writes docs/MODEL_COMPARISON.md
python scripts/risk_report.py
python scripts/simulate_alerts.py
```
`scripts/compare_models.py --quick` checks the pipeline in a short run without using the test lock.

## 21. Testing
```bash
python -m pytest -q                        # 261 tests, about 1.5 minutes
```
Covers data generation, cleaning, features and labels, every model, risk and alert logic, the API (including malformed and hostile input), logging and secret masking, the dashboard client, and integration checks (API = library = offline backtest; no look-ahead; the serving layers import no ML code). Details and the area-by-area map: `docs/TESTING.md`. Measured coverage of the serving layers is 98–100%.

## 22. Docker deployment
```bash
cp .env.example .env                       # optional
docker compose up --build                  # API :8000, dashboard :8501 (both bound to localhost only; no authentication)
docker compose exec api python scripts/smoke_test.py
```
One image runs both services (CPU-only PyTorch, non-root user, health checks). Logs and alert state live in a named volume. **Verified so far:** a user build on Docker succeeded (about 400 s, CPU-only PyTorch), the API container became healthy and the dashboard container started. Not yet confirmed: dashboard behaviour in the container, image size, and the log volume over a restart. Note that `models/` and `reports/` are listed in `.gitignore`: a fresh clone must rebuild them (section 20) or the build fails at `COPY models`. See `docs/PHASES.md` (Phase 18).

## 23. Limitations
- **Simulated data only.** Labels are derived from the same KPIs the generator creates, so the task is easier than a real one. Real networks have drift, outages, planned events and messier KPIs.
- **One dataset.** Seeds vary only the model; results on other simulated datasets or real data may differ.
- **Needs consecutive history per cell** before it can score: 3 hours for the LSTM, about 6 hours for the hybrid (`/model/info` reports the exact number).
- **The online cleaner cannot remove isolated one-sample glitches** (that needs future data). The alert debounce is the defence.
- **Alert trade-off:** the validated threshold favours recall, so false alerts remain (about 0.65 per site-day before the alert engine's de-duplication).
- **Scoring speed:** about 3–5 seconds to score 20 cells, because features are built cell by cell.
- **Serving:** state is in memory in a single process; no authentication, rate limiting or TLS; the API is meant for a trusted network.
- **Docker: the build and startup were confirmed once on one machine; the rest of the container behaviour is untested. The dashboard has not been checked in a real browser by the author.**

## 24. Future enhancements
- Validate on real (or at least anonymized operator) data, and recalibrate.
- Vectorize feature building across cells for much faster batch scoring.
- Longer horizons and multi-horizon output; probabilistic intervals.
- Add neighbouring-cell and handover information (graph or spatial models).
- Drift monitoring and scheduled retraining; a model registry.
- Authentication, rate limiting, and persistent storage (database) for multi-worker deployment.
- Alert feedback (acknowledge or dismiss) to tune thresholds per cell.

## 25. Project structure
```
configs/         config.yaml (single source of truth)
data/sample/     small committed dataset (2 days, 20 cells)
src/             data, preprocessing, features, models, prediction, alerts, api, dashboard, utils
dashboard/       Streamlit app
scripts/         pipeline and utility CLIs (see section 20)
models/          trained artefacts + production_model.json
reports/         evaluation outputs and figures
docs/            FEATURES, SPATIAL_FEATURES, MODEL_COMPARISON, TESTING, PHASES
tests/           261 tests
Dockerfile, docker-compose.yml, requirements*.txt, .env.example
```
