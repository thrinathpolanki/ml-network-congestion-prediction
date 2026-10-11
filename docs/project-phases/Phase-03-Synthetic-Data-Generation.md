# 🚀 Phase 3: Synthetic Data Generation

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Built

| File | Purpose |
| --- | --- |
| `src/data/schema.py` | The data contract: columns, valid ranges, validators |
| `src/data/generator.py` | The synthetic generator |
| `src/data/sources.py` | `DataSource` interface with `SyntheticSource` and `CSVSource`, which is the real-data entry point |
| `src/features/labeling.py` | KPIs → congestion score → 4 states, plus the future target at t+H |
| `scripts/generate_data.py` | CLI for generating data |
| `tests/` | 20 tests for the generator, labeling and config |
| `data/sample/` | Small committed sample (4 sites, 7 days) |

---

## ⚙️ Commands Executed

### Step 1: Generate the default dataset

```powershell
python scripts/generate_data.py
```

**Result:** Generated the `synthetic_network_traffic.csv` and `synthetic_network_traffic.csv.meta.json` files in the `data\raw\` folder.

### Step 2: Generate a dataset with custom options (Parquet)

```powershell
python scripts/generate_data.py --sites 30 --end 2026-04-01 --intensity 1.2 --format parquet
```

**Result:** Generated the `synthetic_network_traffic.parquet` and `synthetic_network_traffic.parquet.meta.json` files in the `data\raw\` folder.
