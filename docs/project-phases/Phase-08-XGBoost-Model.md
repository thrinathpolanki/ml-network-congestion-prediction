# 🚀 Phase 8: XGBoost Model

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `src/models/xgb_model.py` | Training, tuning, class weighting, calibration, alert threshold, SHAP, save/load |
| `scripts/train_xgboost.py` | Search → train → calibrate → threshold → evaluate → figures. `--no-tune` and `--figures-only` are supported |
| `tests/test_xgb.py` | 11 new tests (88 total, all passing) |
| `models/xgb_congestion.json` + `xgb_meta.json` | The model (2.6 MB) and its metadata: feature list, threshold, calibration, parameters, and a "simulated data" note |

Modified: `configs/config.yaml` and `src/models/evaluate.py`

---

## ⚙️ Commands Executed

### Step 1: Install XGBoost

```powershell
python -m pip install xgboost
```

**Result:**

```text
Installing collected packages: xgboost
Successfully installed xgboost-3.4.1
```

### Step 2: Run automated tests

```powershell
python -m pytest -q
```

**Result:**

```text
.............................................................................................................                                           [100%]
109 passed in 24.70s
```

### Step 3: Train the XGBoost model

```powershell
python scripts/train_xgboost.py
```

**Result:**

```text
 trial  max_depth  learning_rate  min_child_weight  class_weight_power  val_pr_auc  best_iteration  seconds
     0          6          0.100                 5               0.500       0.813              80    8.000
     1          4          0.150                 5               0.000       0.814             112    9.100
     2          8          0.050                 1               1.000       0.809              76   13.900
     3          8          0.150                 5               0.500       0.810              40    7.900
     4          6          0.050                20               1.000       0.812             106   11.400
     5          6          0.100                 5               1.000       0.811              45    6.400
     6          4          0.150                20               1.000       0.812              44    5.200
     7          8          0.100                 1               1.000       0.812              43    9.300

Training final model on the full training period...

=== VALIDATION (51,700 rows; SIMULATED DATA; val also used for tuning/threshold => optimistic) ===
                     model  accuracy  macro_f1  cong_precision  cong_recall  pr_auc  recall_new_onset  event_recall  false_alerts_per_site_day
            Majority class     0.856     0.231           0.000        0.000   0.051             0.000         0.000                      0.000
Threshold rule (bandwidth)     0.908     0.641           0.643        0.730   0.746             0.157         0.515                      1.153
   Persistence (state now)     0.914     0.644           0.681        0.681   0.752             0.000         0.398                      0.930
       Seasonal naive (1d)     0.873     0.466           0.406        0.394   0.378             0.333         0.444                      1.097
       Seasonal naive (7d)     0.888     0.486           0.480        0.482   0.455             0.419         0.561                      0.958
          XGBoost (argmax)     0.935     0.689           0.759        0.653   0.817             0.228         0.433                      0.574
       XGBoost (tuned tau)     0.921     0.646           0.544        0.929   0.817             0.779         0.895                      1.064
                     model  macro_f1_ci95 cong_recall_ci95
            Majority class [0.227, 0.235]   [0.000, 0.000]
Threshold rule (bandwidth) [0.607, 0.657]   [0.655, 0.781]
   Persistence (state now) [0.610, 0.659]   [0.598, 0.742]
       Seasonal naive (1d) [0.397, 0.541]   [0.274, 0.552]
       Seasonal naive (7d) [0.448, 0.513]   [0.372, 0.582]
          XGBoost (argmax) [0.650, 0.705]   [0.566, 0.711]
       XGBoost (tuned tau) [0.607, 0.667]   [0.915, 0.939]

Operating points (calibrated P(cong) >= tau):
  tau  precision  recall  new_onset_recall  event_recall  false_alerts_per_site_day
0.100      0.466   0.975             0.923         0.988                      1.170
0.200      0.550   0.923             0.764         0.889                      1.064
0.300      0.619   0.851             0.596         0.795                      0.830
0.400      0.717   0.722             0.342         0.538                      0.557
0.500      0.803   0.597             0.149         0.386                      0.401
0.700      0.912   0.410             0.024         0.140                      0.195

Chosen tau = 0.14 (maximises F2 on validation)
Calibration  raw: ECE 0.0028 Brier 0.0195 | calibrated: ECE 0.0001 Brier 0.0191
Inference: 2.2 ms per 1,000 rows (feature matrix already built)

Top 10 features by mean |SHAP|: bandwidth_utilization_pct, dropped_call_rate_pct, score_now, is_peak_hour, packet_loss_pct, tod_cos, site_p90_util_pct, tod_sin, fc_is_peak_hour, active_users
Gain share by group (%): {'current_kpis': 52.2, 'congestion_history': 15.7, 'ewm': 11.7, 'rolling': 7.2, 'lags': 6.0, 'calendar_now': 2.5, 'calendar_forecast_time': 1.8, 'rate_of_change': 1.1, 'site_static': 1.1, 'site_relative': 0.5, 'region_onehot': 0.3}

Saved model + meta to   ai-network-congestion-prediction\models/ and report to   ai-network-congestion-prediction\reports/xgb_report.json
```
