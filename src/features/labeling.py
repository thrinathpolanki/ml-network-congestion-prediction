"""Congestion definition: measurable KPIs -> congestion score -> 4 states.

States: 0 NORMAL, 1 WARNING, 2 CONGESTED, 3 CRITICAL.
All weights, good/bad bounds and thresholds come from config['labeling'].
The same code labels synthetic and real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import horizon_steps

STATE_NAMES = ["NORMAL", "WARNING", "CONGESTED", "CRITICAL"]
UNKNOWN_ID = -1  # score could not be computed (all indicators missing)


def add_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators that are ratios of raw measurements."""
    out = df.copy()
    users = out["active_users"].clip(lower=1)
    out["dl_throughput_per_user_mbps"] = out["download_throughput_mbps"] / users
    return out


def compute_subscores(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return one 0..1 sub-score column per configured indicator."""
    data = add_derived_indicators(df)
    subs = {}
    for name, p in cfg["labeling"]["indicators"].items():
        if name not in data.columns:
            raise KeyError(f"Indicator '{name}' not found in data.")
        good, bad = float(p["good"]), float(p["bad"])
        if good == bad:
            raise ValueError(f"Indicator '{name}': good and bad must differ.")
        subs[name] = ((data[name] - good) / (bad - good)).clip(0.0, 1.0)
    return pd.DataFrame(subs, index=df.index)


def compute_congestion_score(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Blend of weighted-mean and worst sub-score; NaN-aware (weights renormalised)."""
    lab = cfg["labeling"]
    subs = compute_subscores(df, cfg)
    w = np.array([float(lab["indicators"][c]["weight"]) for c in subs.columns])
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("Indicator weights must be non-negative with a positive sum.")
    arr = subs.to_numpy(dtype=float)
    mask = ~np.isnan(arr)
    weighted = np.where(mask, arr * w, 0.0).sum(axis=1)
    denom = (mask * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        wmean = np.where(denom > 0, weighted / denom, np.nan)
    worst = np.where(mask.any(axis=1), np.nanmax(np.where(mask, arr, -np.inf), axis=1), np.nan)
    b = float(lab.get("max_blend", 0.0))
    return pd.Series((1 - b) * wmean + b * worst, index=df.index, name="congestion_score")


def assign_state_id(score: pd.Series, cfg: dict) -> pd.Series:
    """Map a score to a state id using configured thresholds."""
    t = cfg["labeling"]["thresholds"]
    bounds = [float(t["warning"]), float(t["congested"]), float(t["critical"])]
    if not (bounds[0] < bounds[1] < bounds[2]):
        raise ValueError("Thresholds must satisfy warning < congested < critical.")
    ids = np.digitize(score.to_numpy(dtype=float), bounds)
    ids = np.where(score.isna().to_numpy(), UNKNOWN_ID, ids)
    return pd.Series(ids, index=score.index, name="congestion_state_id", dtype="int64")


def label_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add congestion_score, congestion_state_id and congestion_state (current state)."""
    out = df.copy()
    out["congestion_score"] = compute_congestion_score(out, cfg)
    out["congestion_state_id"] = assign_state_id(out["congestion_score"], cfg)
    names = np.array(STATE_NAMES + ["UNKNOWN"], dtype=object)
    out["congestion_state"] = names[out["congestion_state_id"].replace(UNKNOWN_ID, 4).to_numpy()]
    return out


def add_future_target(df: pd.DataFrame, cfg: dict, steps: int | None = None) -> pd.DataFrame:
    """Attach the state/score at t + horizon, computed per site.

    Requires a regular, sorted time grid per site (guaranteed by preprocessing
    resampling). Rows without a known future (the last `steps` rows of each
    site) or with an unknown future label are dropped.
    """
    steps = steps or horizon_steps(cfg)
    out = df.sort_values(["site_id", "timestamp"]).copy()
    g = out.groupby("site_id", sort=False)
    out["target_state_id"] = g["congestion_state_id"].shift(-steps)
    out["target_score"] = g["congestion_score"].shift(-steps)
    out = out.dropna(subset=["target_state_id"])
    out = out[out["target_state_id"] != UNKNOWN_ID].copy()
    out["target_state_id"] = out["target_state_id"].astype("int64")
    out["target_binary"] = (out["target_state_id"] >= 2).astype("int64")
    return out.reset_index(drop=True)
