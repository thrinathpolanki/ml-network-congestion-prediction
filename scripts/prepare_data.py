"""CLI: clean raw data and show the chronological split.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --input data/sample/sample_network_traffic.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config  # noqa: E402
from src.pipeline.prepare import run_prepare  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean data and report chronological splits.")
    ap.add_argument("--input", help="raw CSV/Parquet (default: data/raw/<raw_filename>)")
    ap.add_argument("--output", help="where to write the cleaned CSV (default: data/processed/<clean_filename>)")
    a = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg)
    r = run_prepare(cfg, a.input, a.output)

    c = r["cleaning"]
    print("\n=== CLEANING REPORT ===")
    for k, v in c.items():
        print(f"  {k:36s}: {v}")
    loc = r["location"]
    print(f"\n=== LOCATION === columns {loc['columns_present']} | rows missing a position: {loc['rows_missing_position']} | "
          f"one fixed position per cell: {loc['cells_with_one_fixed_position']}")
    print("\n=== CHRONOLOGICAL SPLIT (purged by horizon) ===")
    s = r["split"]
    print(f"  boundaries: {s['boundaries']}")
    print(f"  purged    : {s['purged_rows']}")
    for name in ("train", "val", "test"):
        i = s[name]
        print(f"  {name:5s} rows={i['rows']:>8,}  {i['start']} -> {i['end']}")
        print(f"        target distribution (0 NORMAL,1 WARNING,2 CONGESTED,3 CRITICAL): "
              f"{i.get('target_distribution')}")
    print(f"\nClean data: {r['clean_output']}\nFull report: {r['report_file']}")


if __name__ == "__main__":
    main()
