"""CLI: build features + chronological train/val/test sets.

    python scripts/build_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, resolve_path  # noqa: E402
from src.features.dataset import build_dataset, save_bundle  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    cfg = get_config()
    setup_logging(cfg)
    clean = resolve_path(cfg, "processed_dir") / cfg["preprocessing"]["clean_filename"]
    if not clean.exists():
        sys.exit(f"{clean} not found. Run: python scripts/prepare_data.py")
    df = pd.read_csv(clean, parse_dates=["timestamp"])
    t0 = time.time()
    bundle = build_dataset(df, cfg)
    paths = save_bundle(bundle, cfg)
    r = bundle.report
    print(f"\nFeatures: {r['n_features']} | history needed at inference: {r['required_history_steps']} steps")
    for k, v in r["features_per_group"].items():
        print(f"  {k:24s} {v:>3d}")
    print(f"Rows dropped (warm-up / gaps): {r['rows_dropped_incomplete_features']}")
    for n in ("train", "val", "test"):
        i = r["split"][n]
        print(f"  {n:5s} {i['rows']:>8,} rows  {i['start']} -> {i['end']}  target dist: {i['target_distribution']}")
    print(f"\nBuilt in {time.time() - t0:.1f}s. Saved:")
    for k, p in paths.items():
        print(f"  {k:8s} {p}")


if __name__ == "__main__":
    main()
