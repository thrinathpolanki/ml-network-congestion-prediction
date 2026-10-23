# 🚀 Phase 10: Model Comparison

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `src/models/compare.py` | Operating-point policy for any model, operator-curve summary, selection rule, error overlap, latency timing |
| `scripts/compare_models.py` | The full process. `--quick` runs a 1-seed bug check, and `--force` overrides the test lock |
| `scripts/make_comparison_report.py` | Builds `docs/MODEL_COMPARISON.md` from the saved JSON, so no number is hand-copied |
| `tests/test_compare.py` | 11 new tests (114 total, all passing) |
| `models/production_model.json` | The selection decision, with its timestamp |

Modified: `configs/config.yaml`. Generated the full report `docs/MODEL_COMPARISON.md`.

---

## ⚙️ Commands Executed

### Step 1: Run automated tests

```powershell
python -m pytest -q
```

**Result:**

```text
.......................................................................................................................................
135 passed in 58.18s
```

### Step 2: Run the comparison (quick mode)

```powershell
python scripts/compare_models.py --quick
```

**Result:**

```text
=== VALIDATION EVIDENCE & SELECTION (decided BEFORE scoring test) ===
  XGBoost       PR-AUC 0.8132 ± 0.0000 | operator-area 0.499 ± 0.000
  LSTM          PR-AUC 0.7708 ± 0.0000 | operator-area 0.560 ± 0.000
  Hybrid (avg)  PR-AUC 0.8074 ± 0.0000 | operator-area 0.550 ± 0.000
  z-scores vs xgboost {}
  CHOSEN: xgboost - Only 1 seed(s): variance cannot be estimated, so the default (xgboost) is kept.

Scoring TEST set...

=== TEST (51,840 rows; SIMULATED DATA; mean over 1 seeds) ===
                     model macro_f1 pr_auc cong_recall cong_precision recall_new_onset event_recall false_alerts_per_site_day
            Majority class    0.234  0.040       0.000          0.000            0.000        0.000                     0.000
Threshold rule (bandwidth)    0.620  0.698       0.711          0.629            0.157        0.493                     0.739
   Persistence (state now)    0.614  0.700       0.659          0.660            0.000        0.390                     0.483
       Seasonal naive (1d)    0.442  0.319       0.436          0.357            0.379        0.459                     0.978
       Seasonal naive (7d)    0.467  0.424       0.452          0.457            0.359        0.452                     0.761
          XGBoost (argmax)    0.662  0.785       0.649          0.767            0.197        0.418                     0.289
       XGBoost (tuned tau)    0.618  0.785       0.932          0.531            0.805        0.897                     1.000
             LSTM (argmax)    0.626  0.709       0.683          0.654            0.194        0.425                     0.261
          LSTM (tuned tau)    0.586  0.709       0.914          0.479            0.748        0.829                     0.694

95% CIs of the first-seed models (day-block bootstrap):
                     model macro_f1_ci95_seed1 cong_recall_ci95_seed1
            Majority class      [0.229, 0.240]         [0.000, 0.000]
Threshold rule (bandwidth)      [0.588, 0.639]         [0.688, 0.727]
   Persistence (state now)      [0.576, 0.641]         [0.632, 0.676]
       Seasonal naive (1d)      [0.372, 0.490]         [0.262, 0.562]
       Seasonal naive (7d)      [0.419, 0.485]         [0.307, 0.541]
          XGBoost (argmax)      [0.604, 0.701]         [0.561, 0.684]
       XGBoost (tuned tau)      [0.563, 0.659]         [0.890, 0.952]
             LSTM (argmax)      [0.593, 0.649]         [0.614, 0.707]
          LSTM (tuned tau)      [0.547, 0.622]         [0.867, 0.934]

Error overlap (tuned tau, seed 1): {'n_congested': 2057, 'missed_by_a': 139, 'missed_by_b': 177, 'missed_by_both': 93, 'miss_jaccard': 0.417, 'false_alarm_a':1692, 'false_alarm_b': 2041, 'false_alarm_both': 1463, 'false_alarm_jaccard': 0.644, 'caught_by_either': 1964, 'caught_by_a': 1918, 'caught_by_b': 1880}

Per-region (tuned tau, seed 1):
  model            region  rows  congested_rows  recall  precision
XGBoost business_district  7776             333   0.931      0.562
XGBoost             rural 18144               0     NaN        NaN
XGBoost          suburban  5184               6   0.000      0.000
XGBoost        urban_core 12960            1519   0.945      0.538
XGBoost urban_residential  7776             199   0.864      0.460
   LSTM business_district  7776             333   0.865      0.527
   LSTM             rural 18144               0     NaN        NaN
   LSTM          suburban  5184               6   0.000      0.000
   LSTM        urban_core 12960            1519   0.951      0.482
   LSTM urban_residential  7776             199   0.744      0.440

CRITICAL class (tuned tau, seed 1): {'XGBoost': {'precision': 0.556, 'recall': 0.248, 'f1': 0.343, 'support': 222}, 'LSTM': {'precision': 0.316, 'recall': 0.374, 'f1': 0.342, 'support': 222}}

End-to-end latency per single-site request (ms, median / p95): {'xgboost': '69.6 / 74.5', 'lstm': '50.6 / 54.5', 'hybrid': '105.3 / 114.3'}
Complexity: {'xgboost': {'model_file_kb': 1166.5, 'trees_total': 156, 'features': 158, 'history_steps_needed': 73, 'train_seconds_mean': 7.1}, 'lstm': {'parameters': 28548, 'inputs_per_step': 25, 'static_inputs': 13, 'history_steps_needed': 36, 'train_seconds_mean': 6.5}}

Done in 53s. Outputs in  ai-network-congestion-prediction\reports\quick
```

### Step 3: Run the full comparison

```powershell
python scripts/compare_models.py
```

**Result:**

```text
2026-10-15 05:32:59,592 | INFO     | src.features.engineering | FeatureBuilder fitted on 241820 rows, 20 sites, 158 features
2026-10-15 05:33:02,346 | INFO     | src.features.dataset | Dataset built: {'train': 240999, 'val': 51700, 'test': 51840}
2026-10-15 05:33:04,854 | INFO     | src.models.lstm_model | Dropped 1 of 240999 rows (insufficient / incomplete window)
Data ready (7s). Seeds: [42, 43, 44, 45]. Training 8 models...
2026-10-15 05:33:39,262 | INFO     | src.models.lstm_model | Epoch  1 | train 0.5166 | val 0.3946 | val PR-AUC 0.7866 | 9s
2026-10-15 05:33:48,239 | INFO     | src.models.lstm_model | Epoch  2 | train 0.3701 | val 0.3791 | val PR-AUC 0.8014 | 9s
2026-10-15 05:33:56,932 | INFO     | src.models.lstm_model | Epoch  3 | train 0.3581 | val 0.3732 | val PR-AUC 0.8109 | 9s
2026-10-15 05:34:05,615 | INFO     | src.models.lstm_model | Epoch  4 | train 0.3492 | val 0.3726 | val PR-AUC 0.8078 | 9s
2026-10-15 05:34:15,243 | INFO     | src.models.lstm_model | Epoch  5 | train 0.3447 | val 0.3679 | val PR-AUC 0.8126 | 10s
2026-10-15 05:34:24,895 | INFO     | src.models.lstm_model | Epoch  6 | train 0.3435 | val 0.3693 | val PR-AUC 0.8128 | 10s
2026-10-15 05:34:34,557 | INFO     | src.models.lstm_model | Epoch  7 | train 0.3364 | val 0.3767 | val PR-AUC 0.7982 | 10s
2026-10-15 05:34:44,374 | INFO     | src.models.lstm_model | Epoch  8 | train 0.3320 | val 0.3693 | val PR-AUC 0.8101 | 10s
2026-10-15 05:34:54,170 | INFO     | src.models.lstm_model | Epoch  9 | train 0.3319 | val 0.3701 | val PR-AUC 0.8045 | 10s
2026-10-15 05:34:54,171 | INFO     | src.models.lstm_model | Early stopping at epoch 9
  seed 42 done (118s)
2026-10-15 05:35:32,127 | INFO     | src.models.lstm_model | Epoch  1 | train 0.5114 | val 0.3972 | val PR-AUC 0.7890 | 9s
2026-10-15 05:35:42,140 | INFO     | src.models.lstm_model | Epoch  2 | train 0.3649 | val 0.3772 | val PR-AUC 0.8082 | 10s
2026-10-15 05:35:56,454 | INFO     | src.models.lstm_model | Epoch  3 | train 0.3592 | val 0.3723 | val PR-AUC 0.8078 | 14s
2026-10-15 05:36:06,391 | INFO     | src.models.lstm_model | Epoch  4 | train 0.3508 | val 0.3743 | val PR-AUC 0.8038 | 10s
2026-10-15 05:36:16,995 | INFO     | src.models.lstm_model | Epoch  5 | train 0.3507 | val 0.3648 | val PR-AUC 0.8125 | 11s
2026-10-15 05:36:27,720 | INFO     | src.models.lstm_model | Epoch  6 | train 0.3428 | val 0.3722 | val PR-AUC 0.8136 | 11s
2026-10-15 05:36:38,390 | INFO     | src.models.lstm_model | Epoch  7 | train 0.3406 | val 0.3688 | val PR-AUC 0.8077 | 11s
2026-10-15 05:36:48,961 | INFO     | src.models.lstm_model | Epoch  8 | train 0.3320 | val 0.3677 | val PR-AUC 0.8127 | 11s
2026-10-15 05:36:59,803 | INFO     | src.models.lstm_model | Epoch  9 | train 0.3387 | val 0.3634 | val PR-AUC 0.8141 | 11s
2026-10-15 05:37:09,834 | INFO     | src.models.lstm_model | Epoch 10 | train 0.3324 | val 0.3618 | val PR-AUC 0.8157 | 10s
2026-10-15 05:37:21,703 | INFO     | src.models.lstm_model | Epoch 11 | train 0.3298 | val 0.3723 | val PR-AUC 0.8116 | 12s
2026-10-15 05:37:35,132 | INFO     | src.models.lstm_model | Epoch 12 | train 0.3273 | val 0.3642 | val PR-AUC 0.8130 | 13s
2026-10-15 05:37:45,183 | INFO     | src.models.lstm_model | Epoch 13 | train 0.3256 | val 0.3637 | val PR-AUC 0.8171 | 10s
2026-10-15 05:37:56,808 | INFO     | src.models.lstm_model | Epoch 14 | train 0.3246 | val 0.3633 | val PR-AUC 0.8166 | 12s
2026-10-15 05:37:56,808 | INFO     | src.models.lstm_model | Early stopping at epoch 14
  seed 43 done (301s)
2026-10-15 05:38:35,141 | INFO     | src.models.lstm_model | Epoch  1 | train 0.5078 | val 0.3898 | val PR-AUC 0.7883 | 13s
2026-10-15 05:38:46,408 | INFO     | src.models.lstm_model | Epoch  2 | train 0.3668 | val 0.3733 | val PR-AUC 0.8060 | 11s
2026-10-15 05:38:57,779 | INFO     | src.models.lstm_model | Epoch  3 | train 0.3559 | val 0.3684 | val PR-AUC 0.8114 | 11s
2026-10-15 05:39:07,597 | INFO     | src.models.lstm_model | Epoch  4 | train 0.3466 | val 0.3800 | val PR-AUC 0.8080 | 10s
2026-10-15 05:39:17,105 | INFO     | src.models.lstm_model | Epoch  5 | train 0.3487 | val 0.3686 | val PR-AUC 0.8108 | 10s
2026-10-15 05:39:26,461 | INFO     | src.models.lstm_model | Epoch  6 | train 0.3340 | val 0.3701 | val PR-AUC 0.8090 | 9s
2026-10-15 05:39:35,780 | INFO     | src.models.lstm_model | Epoch  7 | train 0.3358 | val 0.3612 | val PR-AUC 0.8180 | 9s
2026-10-15 05:39:45,186 | INFO     | src.models.lstm_model | Epoch  8 | train 0.3327 | val 0.3777 | val PR-AUC 0.8059 | 9s
2026-10-15 05:39:54,612 | INFO     | src.models.lstm_model | Epoch  9 | train 0.3336 | val 0.3656 | val PR-AUC 0.8122 | 9s
2026-10-15 05:40:04,082 | INFO     | src.models.lstm_model | Epoch 10 | train 0.3264 | val 0.3645 | val PR-AUC 0.8127 | 9s
2026-10-15 05:40:13,549 | INFO     | src.models.lstm_model | Epoch 11 | train 0.3260 | val 0.3651 | val PR-AUC 0.8118 | 9s
2026-10-15 05:40:13,550 | INFO     | src.models.lstm_model | Early stopping at epoch 11
  seed 44 done (437s)
2026-10-15 05:40:50,511 | INFO     | src.models.lstm_model | Epoch  1 | train 0.5082 | val 0.3979 | val PR-AUC 0.7830 | 9s
2026-10-15 05:41:00,080 | INFO     | src.models.lstm_model | Epoch  2 | train 0.3723 | val 0.3905 | val PR-AUC 0.7915 | 10s
2026-10-15 05:41:09,616 | INFO     | src.models.lstm_model | Epoch  3 | train 0.3584 | val 0.3734 | val PR-AUC 0.8074 | 10s
2026-10-15 05:41:19,015 | INFO     | src.models.lstm_model | Epoch  4 | train 0.3497 | val 0.3656 | val PR-AUC 0.8197 | 9s
2026-10-15 05:41:28,452 | INFO     | src.models.lstm_model | Epoch  5 | train 0.3466 | val 0.3683 | val PR-AUC 0.8117 | 9s
2026-10-15 05:41:37,883 | INFO     | src.models.lstm_model | Epoch  6 | train 0.3396 | val 0.3727 | val PR-AUC 0.8124 | 9s
2026-10-15 05:41:47,375 | INFO     | src.models.lstm_model | Epoch  7 | train 0.3361 | val 0.3678 | val PR-AUC 0.8085 | 9s
2026-10-15 05:41:56,843 | INFO     | src.models.lstm_model | Epoch  8 | train 0.3336 | val 0.3636 | val PR-AUC 0.8137 | 9s
2026-10-15 05:42:06,295 | INFO     | src.models.lstm_model | Epoch  9 | train 0.3341 | val 0.3661 | val PR-AUC 0.8101 | 9s
2026-10-15 05:42:15,728 | INFO     | src.models.lstm_model | Epoch 10 | train 0.3258 | val 0.3685 | val PR-AUC 0.8109 | 9s
2026-10-15 05:42:25,137 | INFO     | src.models.lstm_model | Epoch 11 | train 0.3280 | val 0.3643 | val PR-AUC 0.8119 | 9s
2026-10-15 05:42:34,580 | INFO     | src.models.lstm_model | Epoch 12 | train 0.3281 | val 0.3661 | val PR-AUC 0.8129 | 9s
2026-10-15 05:42:34,581 | INFO     | src.models.lstm_model | Early stopping at epoch 12
  seed 45 done (578s)

=== VALIDATION EVIDENCE & SELECTION (decided BEFORE scoring test) ===
  XGBoost       PR-AUC 0.8172 ± 0.0007 | operator-area 0.548 ± 0.007
  LSTM          PR-AUC 0.8150 ± 0.0024 | operator-area 0.681 ± 0.023
  Hybrid (avg)  PR-AUC 0.8210 ± 0.0012 | operator-area 0.658 ± 0.020
  z-scores vs xgboost {'lstm': {'pr_auc': -1.2920122115088035, 'operator_area': 7.710694073901014}, 'hybrid': {'pr_auc': 3.772651917885561, 'operator_area': 7.350574304969017}}
  CHOSEN: lstm - lstm beats xgboost by >= 2.0 pooled seed-SDs on at least one validation criterion without being clearly worse on the other.

Scoring TEST set...

=== TEST (51,840 rows; SIMULATED DATA; mean over 4 seeds) ===
                     model      macro_f1        pr_auc   cong_recall cong_precision recall_new_onset  event_recall false_alerts_per_site_day
            Majority class         0.234         0.040         0.000          0.000            0.000         0.000                     0.000
Threshold rule (bandwidth)         0.620         0.698         0.711          0.629            0.157         0.493                     0.739
   Persistence (state now)         0.614         0.700         0.659          0.660            0.000         0.390                     0.483
       Seasonal naive (1d)         0.442         0.319         0.436          0.357            0.379         0.459                     0.978
       Seasonal naive (7d)         0.467         0.424         0.452          0.457            0.359         0.452                     0.761
          XGBoost (argmax) 0.672 ± 0.004 0.792 ± 0.002 0.664 ± 0.007  0.755 ± 0.001    0.267 ± 0.013 0.467 ± 0.003             0.381 ± 0.049
       XGBoost (tuned tau) 0.628 ± 0.009 0.792 ± 0.002 0.929 ± 0.007  0.539 ± 0.013    0.798 ± 0.018 0.885 ± 0.003             0.844 ± 0.047
             LSTM (argmax) 0.672 ± 0.006 0.789 ± 0.001 0.761 ± 0.026  0.713 ± 0.017    0.428 ± 0.061 0.610 ± 0.048             0.283 ± 0.023
          LSTM (tuned tau) 0.630 ± 0.010 0.789 ± 0.001 0.928 ± 0.005  0.535 ± 0.011    0.800 ± 0.017 0.882 ± 0.010             0.681 ± 0.058

95% CIs of the first-seed models (day-block bootstrap):
                     model macro_f1_ci95_seed1 cong_recall_ci95_seed1
            Majority class      [0.229, 0.240]         [0.000, 0.000]
Threshold rule (bandwidth)      [0.588, 0.639]         [0.688, 0.727]
   Persistence (state now)      [0.576, 0.641]         [0.632, 0.676]
       Seasonal naive (1d)      [0.372, 0.490]         [0.262, 0.562]
       Seasonal naive (7d)      [0.419, 0.485]         [0.307, 0.541]
          XGBoost (argmax)      [0.626, 0.713]         [0.584, 0.708]
       XGBoost (tuned tau)      [0.585, 0.672]         [0.873, 0.948]
             LSTM (argmax)      [0.616, 0.712]         [0.688, 0.782]
          LSTM (tuned tau)      [0.573, 0.669]         [0.874, 0.943]

Error overlap (tuned tau, seed 1): {'n_congested': 2057, 'missed_by_a': 156, 'missed_by_b': 162, 'missed_by_both': 122, 'miss_jaccard': 0.622, 'false_alarm_a': 1585, 'false_alarm_b': 1673, 'false_alarm_both': 1424, 'false_alarm_jaccard': 0.776, 'caught_by_either': 1935, 'caught_by_a': 1901, 'caught_by_b': 1895}

Per-region (tuned tau, seed 1):
  model            region  rows  congested_rows  recall  precision
XGBoost business_district  7776             333   0.925      0.578
XGBoost             rural 18144               0     NaN        NaN
XGBoost          suburban  5184               6   0.000      0.000
XGBoost        urban_core 12960            1519   0.939      0.554
XGBoost urban_residential  7776             199   0.834      0.449
   LSTM business_district  7776             333   0.928      0.542
   LSTM             rural 18144               0     NaN        NaN
   LSTM          suburban  5184               6   0.000      0.000
   LSTM        urban_core 12960            1519   0.930      0.550
   LSTM urban_residential  7776             199   0.869      0.412

CRITICAL class (tuned tau, seed 1): {'XGBoost': {'precision': 0.555, 'recall': 0.297, 'f1': 0.387, 'support': 222}, 'LSTM': {'precision': 0.362, 'recall': 0.401, 'f1': 0.38, 'support': 222}}

End-to-end latency per single-site request (ms, median / p95): {'xgboost': '67.9 / 71.7', 'lstm': '48.2 / 56.1', 'hybrid': '108.0 / 124.0'}
Complexity: {'xgboost': {'model_file_kb': 1166.5, 'trees_total': 472, 'features': 158, 'history_steps_needed': 73, 'train_seconds_mean': 24.549999999999997}, 'lstm': {'parameters': 28548, 'inputs_per_step': 25, 'static_inputs': 13, 'history_steps_needed': 36, 'train_seconds_mean': 115.82500000000002}}

Done in 617s. Outputs in  ai-network-congestion-prediction\reports
```
