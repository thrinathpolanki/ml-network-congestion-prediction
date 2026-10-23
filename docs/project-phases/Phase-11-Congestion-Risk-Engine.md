# 🚀 Phase 11: Congestion Risk Engine

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `src/prediction/risk.py` | `RiskEngine` and `RiskAssessment`: probability → risk level, state, confidence, action |
| `src/prediction/history.py` | `prepare_history()`: the online counterpart of offline cleaning, for readings arriving from an API |
| `src/prediction/predictor.py` | `CongestionPredictor`: raw readings → clean history → features → model → assessment, for either model |
| `scripts/risk_report.py` | Checks what the risk levels mean, and checks serving-path parity |
| `tests/test_risk_predictor.py` | 18 new tests (132 total, all passing) |

Overwritten: `configs/config.yaml` (new `serving:` and `risk:` sections), `src/models/lstm_model.py` and `src/models/xgb_model.py`.

---

## ⚙️ Commands Executed

### Step 1: Run automated tests

```powershell
python -m pytest -q 			# expect: 132 passed
```

**Result:**

```text
.......................................................................................................................................                 [100%]
135 passed in 68.21s (0:01:08)
```

### Step 2: Run the risk report

```powershell
python scripts/risk_report.py		# needs generate_data + prepare_data + trained models
```

**Result:**

```text
2026-10-23 22:25:02,235 | INFO     | src.prediction.predictor | Serving model from production_model.json: lstm
2026-10-23 22:25:02,345 | INFO     | src.prediction.predictor | Predictor ready: model=lstm, horizon=6 steps, tau=0.160, history=36 steps
Model: lstm | tau 0.160 | risk thresholds [0.16, 0.4, 0.7]

2026-10-23 22:25:05,281 | INFO     | src.features.engineering | FeatureBuilder fitted on 241820 rows, 20 sites, 158 features
2026-10-23 22:25:10,456 | INFO     | src.features.dataset | Dataset built: {'train': 240999, 'val': 51700, 'test': 51840}
A. Observed congestion frequency by risk level (descriptive; thresholds not tuned)
split risk_level  rows  mean_predicted_prob  observed_congestion_rate
 test        LOW 48272                0.004                     0.003
 test     MEDIUM  1268                0.277                     0.229
 test       HIGH  1391                0.543                     0.578
 test   CRITICAL   909                0.891                     0.880
  val        LOW 47130                0.005                     0.004
  val     MEDIUM  1732                0.278                     0.278
  val       HIGH  1634                0.540                     0.540
  val   CRITICAL  1204                0.906                     0.907
  calibration on test: ECE 0.0019, Brier 0.0153  (out-of-sample)
  calibration on val: ECE 0.0008, Brier 0.0193  (isotonic was fitted on validation: in-sample)

B. Serving-path parity: raw dirty data through the ONLINE preparation vs offline-cleaned data
  rows_compared                    51840
  mean_abs_prob_diff               0.0002
  p99_abs_prob_diff                0.0000
  max_abs_prob_diff                0.3411
  share_rows_diff_gt_0.01          0.0040
  share_rows_diff_gt_0.05          0.0008
  congested_decision_agreement     0.9998
  risk_level_agreement             0.9995
  pr_auc_offline_path              0.7807
  pr_auc_online_path               0.7798
  rows with |diff| > 0.3: 5 (0.01%)
```
