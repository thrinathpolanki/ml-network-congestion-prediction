"""Phase-4 pipeline: raw -> clean -> labelled -> chronological split summary."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import horizon_steps, resolve_path
from src.data.schema import LOCATION_COLUMNS
from src.data.sources import CSVSource
from src.features.labeling import add_future_target, label_dataframe
from src.preprocessing.cleaning import clean_dataframe
from src.preprocessing.splitting import chronological_split

logger = logging.getLogger(__name__)


def run_prepare(cfg: dict, input_path: str | Path | None = None, output_path: str | Path | None = None) -> dict:
    """Clean raw data, save it, and report split statistics. Returns the report.

    Default output is the project's cleaned-data file; pass `output_path` (e.g. when cleaning the sample)
    so that the main cleaned data is never overwritten by accident."""
    raw = Path(input_path) if input_path else resolve_path(cfg, "raw_dir") / cfg["data"]["raw_filename"]
    df = CSVSource(raw, provenance="unknown").load()
    clean, cleaning_report = clean_dataframe(df, cfg)

    default_out = resolve_path(cfg, "processed_dir") / cfg["preprocessing"]["clean_filename"]
    clean_path = Path(output_path) if output_path else default_out
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(clean_path, index=False)

    labelled = add_future_target(label_dataframe(clean, cfg), cfg)
    split = chronological_split(labelled, cfg)
    loc = [c for c in LOCATION_COLUMNS if c in clean.columns]
    location = {"columns_present": loc,
                "rows_missing_position": int(clean[loc].isna().any(axis=1).sum()) if loc else None,
                "cells_with_one_fixed_position": bool((clean.groupby("site_id")["location"].nunique() == 1).all())
                if "location" in clean.columns else None}
    report = {"input": str(raw), "clean_output": str(clean_path),
              "horizon_steps": horizon_steps(cfg), "location": location,
              "cleaning": cleaning_report, "split": split.summary()}

    reports_dir = resolve_path(cfg, "reports_dir")
    name = "preprocessing_report.json" if clean_path == default_out else f"preprocessing_report__{clean_path.stem}.json"
    (reports_dir / name).write_text(json.dumps(report, indent=2, default=str))
    report["report_file"] = str(reports_dir / name)
    logger.info("Preprocessing complete: %s", clean_path)
    return report
