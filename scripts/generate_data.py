"""CLI: generate the synthetic dataset.

    python scripts/generate_data.py
    python scripts/generate_data.py --sites 30 --start 2026-01-01 --end 2026-04-01 \
        --seed 7 --intensity 1.2 --format parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, resolve_path  # noqa: E402
from src.data.generator import SYNTHETIC_WARNING, generate_dataset, save_dataset  # noqa: E402
from src.data.schema import count_invalid_values  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SIMULATED network traffic data.")
    ap.add_argument("--sites", type=int)
    ap.add_argument("--start")
    ap.add_argument("--end", help="exclusive end date")
    ap.add_argument("--freq", help="pandas offset alias, e.g. 5min, 15min")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--intensity", type=float, help="congestion intensity multiplier")
    ap.add_argument("--clean", action="store_true", help="do not inject data-quality issues")
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    ap.add_argument("--output", help="output file path")
    a = ap.parse_args()

    cfg = get_config()
    setup_logging(cfg)
    seed = a.seed if a.seed is not None else cfg["project"]["seed"]
    set_global_seed(seed)

    df = generate_dataset(cfg, n_sites=a.sites, start=a.start, end=a.end, freq=a.freq,
                          seed=seed, intensity=a.intensity,
                          inject_issues=False if a.clean else None)
    stem = Path(cfg["data"]["raw_filename"]).stem
    out = Path(a.output) if a.output else resolve_path(cfg, "raw_dir") / f"{stem}.{a.format}"
    save_dataset(df, out, cfg, seed=seed)

    print(f"\n{SYNTHETIC_WARNING}\n")
    print(f"Saved : {out}")
    print(f"Rows  : {len(df):,} | Sites: {df['site_id'].nunique()} | "
          f"Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Missing cells: {int(df.isna().sum().sum()):,} | "
          f"Duplicate rows: {int(df.duplicated(['site_id', 'timestamp']).sum()):,} | "
          f"Invalid values: {sum(count_invalid_values(df).values())}")


if __name__ == "__main__":
    main()
