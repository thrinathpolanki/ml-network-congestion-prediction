# 🚀 Phase 6: Feature Engineering

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `src/features/engineering.py` | `FeatureBuilder`, the single feature code path for training and the API, plus `SiteBaselines` |
| `src/features/dataset.py` | `build_dataset()`: fit → features → targets → chronological split; `save_bundle()` |
| `scripts/build_features.py` | CLI |
| `docs/FEATURES.md` | Auto-generated feature dictionary with an availability and leakage note per group |
| `tests/test_features.py` | 20 new tests (57 total, all passing) |

Modified: `configs/config.yaml`

---

## ⚙️ Commands Executed

### Step 1: Run automated tests

```powershell
python -m pytest -q
```

**Result:**

```text
..............................................................................                                                       [100%]
78 passed in 20.59s
```

### Step 2: Generate and prepare the data (if not already done)

```powershell
python scripts/generate_data.py && python scripts/prepare_data.py   # if not already done
```

### Step 3: Build the features

```powershell
python scripts/build_features.py
```

**Result:**

```text
Features: 158 | history needed at inference: 73 steps
  current_kpis              15
  lags                      25
  rate_of_change            15
  rolling                   60
  ewm                       10
  congestion_history        10
  calendar_now               7
  calendar_forecast_time     4
  site_static                4
  site_relative              3
  region_onehot              5
Rows dropped (warm-up / gaps): 701
  train  240,999 rows  2026-01-01 02:55:00 -> 2026-02-11 23:00:00  target dist: {0: 0.8654, 1: 0.0902, 2: 0.0379, 3: 0.0065}
  val     51,700 rows  2026-02-11 23:35:00 -> 2026-02-20 22:55:00  target dist: {0: 0.8556, 1: 0.0932, 2: 0.0445, 3: 0.0067}
  test    51,840 rows  2026-02-20 23:30:00 -> 2026-03-01 23:25:00  target dist: {0: 0.8806, 1: 0.0798, 2: 0.0354, 3: 0.0043}

Built in 6.7s. Saved:
  train     ai-network-congestion-prediction\data\processed\features_train.parquet
  val       ai-network-congestion-prediction\data\processed\features_val.parquet
  test      ai-network-congestion-prediction\data\processed\features_test.parquet
  builder   ai-network-congestion-prediction\models\feature_builder.joblib
  report    ai-network-congestion-prediction\reports\feature_report.json
  docs      ai-network-congestion-prediction\docs\FEATURES.md
```
