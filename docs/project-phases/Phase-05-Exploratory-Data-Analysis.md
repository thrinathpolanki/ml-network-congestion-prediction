# 🚀 Phase 5: Exploratory Data Analysis

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `src/analysis/eda.py` | The nine figures and a machine-readable `eda_summary.json` |
| `scripts/run_eda.py` | CLI entry point |
| `notebooks/01_eda.ipynb` | Thin notebook that calls the same code, so there is one source of truth |
| `tests/test_eda.py` | 2 new tests (37 total, all passing) |

---

## ⚙️ Commands Executed

### Step 1: Install Matplotlib

```powershell
python -m pip install matplotlib
```

**Result:** The Matplotlib library was installed.

### Step 2: Run automated tests

```powershell
python -m pytest -q
```

**Result:**

```text
......................................................                                                         [100%]
54 passed in 22.80s
```

### Step 3: Run the EDA script

```powershell
python scripts/run_eda.py
```

**Result:**

```text
Figures:
  01_timeseries_kpis.png
  02_congestion_distribution.png
  03_congestion_by_hour_day.png
  04_congestion_by_site.png
  05_correlation.png
  06_peak_hour_behavior.png
  07_feature_distributions.png
  08_temporal_structure.png
  09_onset_trajectory.png
  09a_cell_map.png
  09b_sample_day_heatmap.png

Summary written to reports/figures/eda_summary.json (SIMULATED DATA, training period only)
```
