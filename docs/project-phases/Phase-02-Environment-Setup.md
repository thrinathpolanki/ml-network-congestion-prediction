# 🚀 Phase 2: Environment Setup

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Built

| File | Purpose |
| --- | --- |
| `configs/config.yaml` | All thresholds, weights, horizon, generator and region settings |
| `src/config.py` | YAML loader with `NCP_SECTION__KEY` env overrides, `.env` support, `horizon_steps()` |
| `src/utils/seed.py`, `logging_config.py` | Global seeding; console and rotating-file logging with secret redaction |
| `requirements.txt`, `.gitignore`, `.env.example` | Project files |

---

## ⚙️ Commands Executed

### Step 1: Create and activate the virtual environment

```powershell
python --version
python -m venv .venv
Test-Path .venv\Scripts\Activate.ps1
.venv\Scripts\Activate.ps1
```

**Result:** Created the `.venv` folder.

### Step 2: Install required libraries

```powershell
pip install numpy pandas scipy pyyaml python-dotenv pyarrow pytest
```

**Result:** Installed the required libraries.
