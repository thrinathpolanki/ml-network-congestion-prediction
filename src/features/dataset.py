"""Build the supervised dataset: features at t -> congestion state at t + horizon."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import resolve_path
from src.features.engineering import KEYS, FeatureBuilder
from src.features.labeling import add_derived_indicators, add_future_target, label_dataframe
from src.preprocessing.splitting import SplitResult, chronological_split

logger = logging.getLogger(__name__)
TARGET_COLUMNS = ["target_state_id", "target_binary", "target_score"]


@dataclass
class DatasetBundle:
    builder: FeatureBuilder
    split: SplitResult
    report: dict

    @property
    def train(self): return self.split.train
    @property
    def val(self): return self.split.val
    @property
    def test(self): return self.split.test


def build_dataset(clean: pd.DataFrame, cfg: dict) -> DatasetBundle:
    """clean (from clean_dataframe) -> fitted FeatureBuilder + chronological splits.

    Order matters for leakage control:
      1. choose split boundaries from the labelled timeline
      2. fit the builder ONLY on rows before the train/validation boundary
      3. build features for the whole timeline (features only look backwards)
      4. attach targets (state at t+H), drop incomplete rows, split with purging
    """
    labelled = add_future_target(label_dataframe(add_derived_indicators(clean), cfg), cfg)
    s0 = chronological_split(labelled, cfg)
    t1 = pd.Timestamp(s0.boundaries["train_end_exclusive"])
    t2 = pd.Timestamp(s0.boundaries["val_end_exclusive"])

    builder = FeatureBuilder(cfg).fit(clean[clean["timestamp"] < t1])
    feats = builder.transform(clean, drop_incomplete=False)
    data = feats.merge(labelled[KEYS + TARGET_COLUMNS], on=KEYS, how="inner")
    complete = data[builder.feature_columns].notna().all(axis=1)
    n_dropped = int((~complete).sum())
    data = data.loc[complete].reset_index(drop=True)

    split = chronological_split(data, cfg, boundaries=(t1, t2))
    report = {
        "n_features": len(builder.feature_columns),
        "features_per_group": {k: len(v) for k, v in builder.feature_groups.items()},
        "required_history_steps": builder.required_history_steps,
        "rows_dropped_incomplete_features": n_dropped,
        "baselines_fitted_on_rows_before": str(t1),
        "split": split.summary(),
    }
    logger.info("Dataset built: %s", {k: len(getattr(split, k)) for k in ("train", "val", "test")})
    return DatasetBundle(builder, split, report)


def save_bundle(bundle: DatasetBundle, cfg: dict) -> dict[str, Path]:
    """Write parquet splits, the fitted builder, a JSON report and docs/FEATURES.md."""
    out_dir = resolve_path(cfg, "processed_dir")
    prefix = cfg["features"]["processed_prefix"]
    paths: dict[str, Path] = {}
    for name in ("train", "val", "test"):
        p = out_dir / f"{prefix}_{name}.parquet"
        getattr(bundle, name).to_parquet(p, index=False)
        paths[name] = p
    paths["builder"] = bundle.builder.save(resolve_path(cfg, "models_dir") / cfg["features"]["builder_filename"])
    paths["report"] = resolve_path(cfg, "reports_dir") / "feature_report.json"
    paths["report"].write_text(json.dumps(bundle.report, indent=2, default=str))
    docs = Path(__file__).resolve().parents[2] / "docs"
    docs.mkdir(exist_ok=True)
    paths["docs"] = docs / "FEATURES.md"
    paths["docs"].write_text(bundle.builder.describe(), encoding="utf-8")
    return paths
