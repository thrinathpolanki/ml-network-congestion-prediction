"""CLI: exploratory analysis on the TRAINING period of the cleaned data.

    python scripts/run_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.eda import fig_sample_heatmap, run_eda  # noqa: E402
from src.config import get_config, resolve_path  # noqa: E402
from src.features.labeling import add_future_target, label_dataframe  # noqa: E402
from src.preprocessing.splitting import chronological_split  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    cfg = get_config()
    setup_logging(cfg)
    clean = resolve_path(cfg, "processed_dir") / cfg["preprocessing"]["clean_filename"]
    if not clean.exists():
        sys.exit(f"{clean} not found. Run: python scripts/prepare_data.py")
    df = pd.read_csv(clean, parse_dates=["timestamp"])
    train = chronological_split(add_future_target(label_dataframe(df, cfg), cfg), cfg).train
    s = run_eda(train, cfg, resolve_path(cfg, "reports_dir") / "figures")
    sample = resolve_path(cfg, "sample_dir") / cfg["sample"]["filename"]
    if sample.exists():                                       # the clean 20-cell sample: complete 5-minute grid for every cell
        s["figures"]["sample_heatmap"] = fig_sample_heatmap(pd.read_csv(sample, parse_dates=["timestamp"]), resolve_path(cfg, "reports_dir") / "figures")
    print("\nFigures:", *[Path(p).name for p in s["figures"].values()], sep="\n  ")
    print("\nSummary written to reports/figures/eda_summary.json (SIMULATED DATA, training period only)")


if __name__ == "__main__":
    main()
