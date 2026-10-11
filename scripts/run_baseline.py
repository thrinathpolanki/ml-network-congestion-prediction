"""CLI: fit + evaluate all baselines on the validation and test sets.

    python scripts/run_baselines.py
Needs: python scripts/build_features.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, horizon_steps, resolve_path  # noqa: E402
from src.models.baselines import (KEYS, MajorityClass, Persistence, SeasonalNaive,  # noqa: E402
                                  ThresholdDetector)
from src.models.evaluate import build_frame, compare_frames, plot_confusions  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

SHOW = ["model", "accuracy", "macro_f1", "cong_precision", "cong_recall", "cong_f1", "pr_auc",
        "recall_new_onset", "event_recall", "false_alerts_per_site_day"]


def main() -> None:
    cfg = get_config()
    setup_logging(cfg)
    d = resolve_path(cfg, "processed_dir")
    pre = cfg["features"]["processed_prefix"]
    paths = {s: d / f"{pre}_{s}.parquet" for s in ("train", "val", "test")}
    if not all(p.exists() for p in paths.values()):
        sys.exit("Feature files missing. Run: python scripts/build_features.py")
    train, val, test = (pd.read_parquet(paths[s]) for s in ("train", "val", "test"))
    hist = pd.concat([train, val, test])[KEYS + ["state_now", "score_now"]]
    h, step = horizon_steps(cfg), pd.Timedelta(cfg["data"]["sampling_freq"])

    models = [MajorityClass(), ThresholdDetector(cfg["baselines"]["threshold_feature"]), Persistence()]
    models += [SeasonalNaive(int(k), h, step).set_history(hist) for k in cfg["baselines"]["seasonal_days"]]
    for m in models:
        m.fit(train)

    reports = resolve_path(cfg, "reports_dir")
    out = {}
    for split, df in (("val", val), ("test", test)):
        frames = {m.name: build_frame(df, m.predict(df), m.congestion_score(df)) for m in models}
        table, results, aligned = compare_frames(frames, cfg)
        out[split] = results
        table.to_csv(reports / f"baseline_comparison_{split}.csv", index=False)
        pred_dir = resolve_path(cfg, "reports_dir") / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for name, f in aligned.items():
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            f.to_parquet(pred_dir / f"{split}__{slug}.parquet", index=False)
        print(f"\n=== BASELINES on {split.upper()} ({results[models[0].name]['n_rows']:,} rows; SIMULATED DATA) ===")
        print(table[SHOW].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("95% CI (day-block bootstrap):")
        print(table[["model", "macro_f1_ci95", "cong_recall_ci95"]].to_string(index=False))
        if split == "test":
            plot_confusions(results, reports / "figures" / "10_baseline_confusions_test.png",
                            "Baselines on TEST (rows = true state, simulated data)")
    t = next(m for m in models if isinstance(m, ThresholdDetector))
    print(f"\nFitted utilisation cut-offs (Warning / Congested / Critical): {[round(float(c), 1) for c in t.cuts_]}")
    (reports / "baseline_results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved to {reports}/baseline_*.csv|json")


if __name__ == "__main__":
    main()
