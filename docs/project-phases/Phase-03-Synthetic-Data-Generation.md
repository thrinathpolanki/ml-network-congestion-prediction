# 🚀 Phase 3: Synthetic Data Generation

**Project:** Network Congestion Prediction & Alert System

---

## ⚙️ Commands Executed

### Step 1: Generate default dataset

```powershell
python scripts/generate_data.py
```

**Result:** Generated the `synthetic_network_traffic.csv` and `synthetic_network_traffic.csv.meta.json` files in the `data\raw\` folder.

### Step 2: Generate dataset with custom options

```powershell
python scripts/generate_data.py --sites 30 --end 2026-04-01 --intensity 1.2 --format parquet
```

**Result:** Generated the `synthetic_network_traffic.parquet` and `synthetic_network_traffic.parquet.meta.json` files in the `data\raw\` folder.
