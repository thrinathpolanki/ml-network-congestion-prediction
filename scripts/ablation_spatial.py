"""Does cell POSITION help prediction?  Validation-only ablation (the test set is not touched).

Variants (same XGBoost parameters, same seeds, early stopping on validation):
  A  baseline                      the 158 production features
  B  + static position             latitude, longitude, distance from the map centre
  C  + neighbour load (current)    mean / max current utilisation of the 3 nearest OTHER cells at the same timestamp
                                   (the classic 'spatial' feature; at serving time it needs the neighbours' latest readings)

    python scripts/ablation_spatial.py        # ~10 min on one core; writes reports/spatial_ablation.json
Needs: scripts/prepare_data.py and a trained models/feature_builder.joblib (scripts/build_features.py).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.eda import _haversine_km  # noqa: E402
from src.config import get_config, resolve_path  # noqa: E402
from src.features.dataset import build_dataset  # noqa: E402
from src.models.xgb_model import XGBCongestionModel  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

KEYS = ["site_id", "timestamp"]


def add_spatial_columns(clean: pd.DataFrame, cfg: dict, k: int = 3) -> pd.DataFrame:
    """Return (site, timestamp) -> lat, lon, dist_km, nbr_util_mean3, nbr_util_max3 for every row of `clean`."""
    loc = cfg["generator"]["location"]
    cells = clean.groupby("site_id")[["latitude", "longitude"]].first()
    cells["dist_km"] = _haversine_km(float(loc["center_lat"]), float(loc["center_lon"]), cells.latitude, cells.longitude)
    ids = list(cells.index)
    D = np.array([[_haversine_km(cells.latitude[a], cells.longitude[a], cells.latitude[b], cells.longitude[b]) for b in ids] for a in ids])
    np.fill_diagonal(D, np.inf)
    nearest = np.argsort(D, axis=1)[:, :k]                                      # the k nearest OTHER cells, per cell
    util = clean.pivot_table(index="timestamp", columns="site_id", values="bandwidth_utilization_pct")[ids]
    u = util.to_numpy()
    mean3 = np.stack([np.nanmean(u[:, nearest[i]], axis=1) for i in range(len(ids))], axis=1)
    max3 = np.stack([np.nanmax(u[:, nearest[i]], axis=1) for i in range(len(ids))], axis=1)
    long = pd.concat([pd.DataFrame(mean3, index=util.index, columns=ids).stack().rename("nbr_util_mean3"),
                      pd.DataFrame(max3, index=util.index, columns=ids).stack().rename("nbr_util_max3")], axis=1).reset_index()
    long.columns = ["timestamp", "site_id", "nbr_util_mean3", "nbr_util_max3"]
    long = long.merge(cells.reset_index(), on="site_id")
    return long[KEYS + ["latitude", "longitude", "dist_km", "nbr_util_mean3", "nbr_util_max3"]]


def main() -> None:
    cfg = get_config()
    setup_logging(cfg)
    clean_path = resolve_path(cfg, "processed_dir") / cfg["preprocessing"]["clean_filename"]
    if not clean_path.exists():
        sys.exit("Run scripts/prepare_data.py first.")
    clean = pd.read_csv(clean_path, parse_dates=["timestamp"])
    b = build_dataset(clean, cfg)
    sp = add_spatial_columns(clean, cfg)
    train, val = (d.merge(sp, on=KEYS, how="left") for d in (b.train, b.val))
    assert train[["latitude", "nbr_util_mean3"]].notna().all().all()
    base_cols = b.builder.feature_columns
    variants = {"A baseline (158 features)": base_cols,
                "B + position (lat, lon, distance)": base_cols + ["latitude", "longitude", "dist_km"],
                "C + neighbour load (3 nearest cells)": base_cols + ["nbr_util_mean3", "nbr_util_max3"]}
    params = json.loads((resolve_path(cfg, "models_dir") / cfg["xgboost"]["meta_filename"]).read_text())["params"]
    yb = val["target_state_id"].to_numpy() >= 2
    res = {name: [] for name in variants}
    seeds = [42, 43, 44]
    for s in seeds:
        c = copy.deepcopy(cfg); c["project"]["seed"] = s
        for name, cols in variants.items():
            m = XGBCongestionModel(c).fit(train, val, cols, params)
            ap = float(average_precision_score(yb, m.predict_proba(val)[:, 2:].sum(1)))
            res[name].append(ap)
            print(f"seed {s} | {name:40s} val PR-AUC {ap:.4f} | trees {m.meta['best_iteration']}", flush=True)
    base = np.array(res["A baseline (158 features)"])
    out = {"seeds": seeds, "val_pr_auc": res, "summary": {}}
    print("\nVALIDATION PR-AUC (mean +- sd over seeds; test set untouched)")
    for name, v in res.items():
        v = np.array(v)
        diff = v - base
        out["summary"][name] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "mean_diff_vs_A": float(diff.mean()),
                                "diff_per_seed": diff.round(5).tolist()}
        print(f"  {name:40s} {v.mean():.4f} +- {v.std(ddof=1):.4f}   vs A: {diff.mean():+.4f}  (per seed {np.round(diff, 4).tolist()})")
    (resolve_path(cfg, "reports_dir") / "spatial_ablation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
