# 🚀 Phase 9: LSTM Model

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `src/models/lstm_model.py` | Sequence builder, PyTorch network, training with early stopping and checkpointing, calibration, threshold, save/load |
| `scripts/train_lstm.py` | Train, then evaluate next to XGBoost. `--quick` and `--figures-only` are supported |
| `tests/test_lstm.py` | 15 new tests (103 total, all passing) |
| `models/lstm_congestion.pt`, `lstm_meta.json`, `lstm_scaler.joblib` | Weights, metadata and scaler |

Modified: `configs/config.yaml`

---

## ⚙️ Commands Executed

### Step 1: Install PyTorch

```powershell
python -m pip install torch
```

**Result:** Installed the torch library.

### Step 2: Run automated tests

```powershell
python -m pytest -q
```

**Result:**

```text
.............................................................................................................                                           [100%]
109 passed, 1 skipped in 24.30s
```

### Step 3: Train the LSTM (quick mode)

```powershell
python scripts/train_lstm.py --quick
```

**Result:**

```text
2026-10-12 22:06:30,009 | INFO     | src.features.engineering | FeatureBuilder fitted on 241820 rows, 20 sites, 158 features
2026-10-12 22:06:34,737 | INFO     | src.features.dataset | Dataset built: {'train': 240999, 'val': 51700, 'test': 51840}
Features ready in 12s
2026-10-12 22:06:39,408 | INFO     | src.models.lstm_model | Dropped 1 of 240999 rows (insufficient / incomplete window)
Windows: train 240,998 | val 51,700 | seq_len 36 | 25 dynamic + 13 static inputs | 28,548 parameters

Training (early stopping on weighted validation loss)...
2026-10-12 22:06:59,333 | INFO     | src.models.lstm_model | Epoch  1 | train 0.9586 | val 0.5963 | val PR-AUC 0.6884 | 4s
2026-10-12 22:07:03,738 | INFO     | src.models.lstm_model | Epoch  2 | train 0.5165 | val 0.4579 | val PR-AUC 0.7562 | 4s
2026-10-12 22:07:07,916 | INFO     | src.models.lstm_model | Epoch  3 | train 0.4283 | val 0.4210 | val PR-AUC 0.7708 | 4s
 epoch  train_loss  val_loss  val_pr_auc     lr  seconds
     1      0.9586    0.5963      0.6884 0.0010   4.5000
     2      0.5165    0.4579      0.7562 0.0010   4.4000
     3      0.4283    0.4210      0.7708 0.0010   4.2000

=== VALIDATION, common rows (51,700; SIMULATED DATA; val used for early stopping/calibration/tau => optimistic) ===
                     model  accuracy  macro_f1  cong_precision  cong_recall  pr_auc  recall_new_onset  event_recall  false_alerts_per_site_day
            Majority class     0.856     0.231           0.000        0.000   0.051             0.000         0.000                      0.000
Threshold rule (bandwidth)     0.908     0.641           0.643        0.730   0.746             0.157         0.515                      1.153
   Persistence (state now)     0.914     0.644           0.681        0.681   0.752             0.000         0.398                      0.930
          XGBoost (argmax)     0.935     0.689           0.759        0.653   0.817             0.228         0.433                      0.574
       XGBoost (tuned tau)     0.921     0.646           0.544        0.929   0.817             0.779         0.895                      1.064
             LSTM (argmax)     0.911     0.648           0.679        0.698   0.771             0.174         0.450                      0.524
          LSTM (tuned tau)     0.895     0.605           0.487        0.922   0.771             0.755         0.854                      0.808
                     model  macro_f1_ci95 cong_recall_ci95
            Majority class [0.227, 0.235]   [0.000, 0.000]
Threshold rule (bandwidth) [0.607, 0.657]   [0.655, 0.781]
   Persistence (state now) [0.610, 0.659]   [0.598, 0.742]
          XGBoost (argmax) [0.650, 0.705]   [0.566, 0.711]
       XGBoost (tuned tau) [0.607, 0.667]   [0.915, 0.939]
             LSTM (argmax) [0.625, 0.657]   [0.620, 0.751]
          LSTM (tuned tau) [0.578, 0.619]   [0.899, 0.936]

Operating points (calibrated P(cong) >= tau), LSTM:
  tau  precision  recall  new_onset_recall  event_recall  false_alerts_per_site_day
0.050      0.385   0.978             0.933         0.971                      0.925
0.100      0.446   0.951             0.847         0.901                      0.925
0.150      0.484   0.925             0.766         0.854                      0.791
0.200      0.506   0.905             0.702         0.819                      0.746
0.300      0.616   0.791             0.373         0.579                      0.590
0.400      0.674   0.711             0.192         0.456                      0.512
0.500      0.757   0.587             0.031         0.240                      0.401
0.600      0.808   0.504             0.007         0.146                      0.273
0.700      0.882   0.349             0.000         0.053                      0.134
0.800      0.921   0.281             0.000         0.029                      0.072
0.900      0.957   0.185             0.000         0.012                      0.017

Chosen tau = 0.16
Inference: 18 ms per 1,000 windows (includes window gathering)

Saved to   ai-network-congestion-prediction\models/ and   ai-network-congestion-prediction\reports/lstm_report.json  (total 52s)
```

### Step 4: Train the LSTM (full)

```powershell
python scripts/train_lstm.py
```

**Result:**

```text
2026-10-12 22:08:16,810 | INFO     | src.features.engineering | FeatureBuilder fitted on 241820 rows, 20 sites, 158 features
2026-10-12 22:08:19,178 | INFO     | src.features.dataset | Dataset built: {'train': 240999, 'val': 51700, 'test': 51840}
Features ready in 6s
2026-10-12 22:08:21,416 | INFO     | src.models.lstm_model | Dropped 1 of 240999 rows (insufficient / incomplete window)
Windows: train 240,998 | val 51,700 | seq_len 36 | 25 dynamic + 13 static inputs | 28,548 parameters

Training (early stopping on weighted validation loss)...
2026-10-12 22:08:30,093 | INFO     | src.models.lstm_model | Epoch  1 | train 0.5166 | val 0.3946 | val PR-AUC 0.7866 | 8s
2026-10-12 22:08:37,730 | INFO     | src.models.lstm_model | Epoch  2 | train 0.3701 | val 0.3791 | val PR-AUC 0.8014 | 8s
2026-10-12 22:08:46,974 | INFO     | src.models.lstm_model | Epoch  3 | train 0.3581 | val 0.3732 | val PR-AUC 0.8109 | 9s
2026-10-12 22:08:55,915 | INFO     | src.models.lstm_model | Epoch  4 | train 0.3492 | val 0.3726 | val PR-AUC 0.8078 | 9s
2026-10-12 22:09:04,322 | INFO     | src.models.lstm_model | Epoch  5 | train 0.3447 | val 0.3679 | val PR-AUC 0.8126 | 8s
2026-10-12 22:09:12,759 | INFO     | src.models.lstm_model | Epoch  6 | train 0.3435 | val 0.3693 | val PR-AUC 0.8128 | 8s
2026-10-12 22:09:21,277 | INFO     | src.models.lstm_model | Epoch  7 | train 0.3364 | val 0.3767 | val PR-AUC 0.7982 | 9s
2026-10-12 22:09:29,786 | INFO     | src.models.lstm_model | Epoch  8 | train 0.3320 | val 0.3693 | val PR-AUC 0.8101 | 9s
2026-10-12 22:09:38,397 | INFO     | src.models.lstm_model | Epoch  9 | train 0.3319 | val 0.3701 | val PR-AUC 0.8045 | 9s
2026-10-12 22:09:38,398 | INFO     | src.models.lstm_model | Early stopping at epoch 9
 epoch  train_loss  val_loss  val_pr_auc     lr  seconds
     1      0.5166    0.3946      0.7866 0.0010   7.6000
     2      0.3701    0.3791      0.8014 0.0010   7.6000
     3      0.3581    0.3732      0.8109 0.0010   9.2000
     4      0.3492    0.3726      0.8078 0.0010   8.9000
     5      0.3447    0.3679      0.8126 0.0010   8.4000
     6      0.3435    0.3693      0.8128 0.0010   8.4000
     7      0.3364    0.3767      0.7982 0.0005   8.5000
     8      0.3320    0.3693      0.8101 0.0005   8.5000
     9      0.3319    0.3701      0.8045 0.0003   8.6000

=== VALIDATION, common rows (51,700; SIMULATED DATA; val used for early stopping/calibration/tau => optimistic) ===
                     model  accuracy  macro_f1  cong_precision  cong_recall  pr_auc  recall_new_onset  event_recall  false_alerts_per_site_day
            Majority class     0.856     0.231           0.000        0.000   0.051             0.000         0.000                      0.000
Threshold rule (bandwidth)     0.908     0.641           0.643        0.730   0.746             0.157         0.515                      1.153
   Persistence (state now)     0.914     0.644           0.681        0.681   0.752             0.000         0.398                      0.930
          XGBoost (argmax)     0.935     0.689           0.759        0.653   0.817             0.228         0.433                      0.574
       XGBoost (tuned tau)     0.921     0.646           0.544        0.929   0.817             0.779         0.895                      1.064
             LSTM (argmax)     0.924     0.685           0.710        0.716   0.813             0.350         0.497                      0.468
          LSTM (tuned tau)     0.910     0.649           0.537        0.927   0.813             0.782         0.895                      0.752
                     model  macro_f1_ci95 cong_recall_ci95
            Majority class [0.227, 0.235]   [0.000, 0.000]
Threshold rule (bandwidth) [0.607, 0.657]   [0.655, 0.781]
   Persistence (state now) [0.610, 0.659]   [0.598, 0.742]
          XGBoost (argmax) [0.650, 0.705]   [0.566, 0.711]
       XGBoost (tuned tau) [0.607, 0.667]   [0.915, 0.939]
             LSTM (argmax) [0.646, 0.706]   [0.628, 0.772]
          LSTM (tuned tau) [0.615, 0.668]   [0.913, 0.935]

Operating points (calibrated P(cong) >= tau), LSTM:
  tau  precision  recall  new_onset_recall  event_recall  false_alerts_per_site_day
0.050      0.435   0.986             0.955         0.982                      0.791
0.100      0.472   0.970             0.907         0.953                      0.758
0.150      0.525   0.936             0.805         0.901                      0.707
0.200      0.563   0.903             0.720         0.865                      0.758
0.300      0.629   0.841             0.567         0.766                      0.585
0.400      0.696   0.746             0.389         0.591                      0.474
0.500      0.791   0.609             0.182         0.380                      0.323
0.600      0.832   0.542             0.123         0.281                      0.267
0.700      0.907   0.413             0.040         0.152                      0.156
0.800      0.924   0.378             0.024         0.117                      0.111
0.900      0.960   0.287             0.001         0.058                      0.050

Chosen tau = 0.16
Inference: 16 ms per 1,000 windows (includes window gathering)

Saved to   ai-network-congestion-prediction\models/ and   ai-network-congestion-prediction\reports/lstm_report.json  (total 89s)
```
