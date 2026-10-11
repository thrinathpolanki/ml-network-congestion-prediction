# 🚀 Phase 4: Data Preprocessing

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `src/preprocessing/cleaning.py` | The full cleaning pipeline, returning a report |
| `src/preprocessing/splitting.py` | Chronological split with target purging |
| `src/preprocessing/scaling.py` | `FeatureTransformer`: scaling and one-hot encoding, fitted on train only, saved with joblib |
| `src/pipeline/prepare.py`, `scripts/prepare_data.py` | End-to-end run: clean → label → split → report |
| `tests/test_preprocessing.py` | 15 new tests (35 total) |
| `configs/config.yaml` | Updated with `preprocessing:` settings |

---

## ⚙️ Commands Executed

### Step 1: Install scikit-learn

```powershell
python -m pip install scikit-learn
```

**Result:** The scikit-learn library was installed.

### Step 2: Check scikit-learn version

```powershell
python -c "import sklearn; print(sklearn.__version__)"
```

**Result:** The command prints the installed scikit-learn version.

### Step 3: Run automated tests

```powershell
python -m pytest -q
```

**Result:**

```text
................................................                                                               [100%]
48 passed
```

**Explanation:** This command checks whether my code behaves as expected across the automated tests, including data preprocessing and other tested functionality.

> ⚠️ **Important:** Passing all 48 tests does not prove that the ML model is accurate or ready for production. It means only that the tested behaviors passed.

### Step 4: Run the data preprocessing script

```powershell
python scripts/prepare_data.py
```

**Result:**

- Generated the clean data file `clean_network_traffic.csv` in the `data\processed\` folder.
- Generated the full report `preprocessing_report.json` in the `reports\` folder.
