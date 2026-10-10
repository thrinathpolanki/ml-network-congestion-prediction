"""The small committed SAMPLE dataset: N cells, every cell on a perfect 5-minute grid.

Unlike the main training data, the sample is deliberately clean: no gaps, duplicates or missing values,
so every cell has exactly the same, complete timeline. It is SIMULATED data.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.data.generator import generate_dataset
from src.data.schema import LOCATION_COLUMNS

LOCATION_RE = re.compile(r"^-?\d{1,2}\.\d{6},-?\d{1,3}\.\d{6}$")


def build_sample(cfg: dict) -> pd.DataFrame:
    """Generate the sample from config['sample'] (same simulator, issue injection switched off)."""
    s = cfg["sample"]
    return generate_dataset(cfg, n_sites=int(s["n_sites"]), start=s["start"], end=s["end"], inject_issues=False)


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def validate_sample(df: pd.DataFrame, cfg: dict) -> dict:
    """Check every guarantee of the sample; returns {check: (passed, detail)}."""
    s, loc = cfg["sample"], cfg["generator"]["location"]
    step = pd.Timedelta(cfg["data"]["sampling_freq"])
    expected_ts = pd.date_range(s["start"], s["end"], freq=cfg["data"]["sampling_freq"], inclusive="left")
    g = df.groupby("site_id")
    gaps = df.sort_values(["site_id", "timestamp"]).groupby("site_id")["timestamp"].diff().dropna()
    per_cell = g["timestamp"].apply(lambda t: bool((t.sort_values().to_numpy() == expected_ts.to_numpy()).all()))
    first = g[LOCATION_COLUMNS + ["region"]].first()
    dist = _haversine_km(float(loc["center_lat"]), float(loc["center_lon"]), first["latitude"].to_numpy(), first["longitude"].to_numpy())
    max_r = np.array([float(loc["radius_km"][r]) for r in first["region"]])
    fmt_ok = df["location"].map(lambda v: bool(LOCATION_RE.match(v))).all() and \
        (df["location"] == df["latitude"].map("{:.6f}".format) + "," + df["longitude"].map("{:.6f}".format)).all()
    return {
        "cells": (df["site_id"].nunique() == int(s["n_sites"]), f"{df['site_id'].nunique()} cells"),
        "rows_per_cell": (bool((g.size() == len(expected_ts)).all()), f"{len(expected_ts)} rows each"),
        "every_cell_has_complete_timeline": (bool(per_cell.all()), "timestamps equal the full expected grid"),
        "all_gaps_exactly_one_step": (bool((gaps == step).all()), f"gap = {step}"),
        "no_duplicates": (not df.duplicated(["site_id", "timestamp"]).any(), ""),
        "no_missing_values": (int(df.isna().sum().sum()) == 0, f"{int(df.isna().sum().sum())} missing"),
        "location_format": (bool(fmt_ok), 'e.g. "' + str(df["location"].iloc[0]) + '"'),
        "one_fixed_position_per_cell": (bool((g["location"].nunique() == 1).all()), ""),
        "distinct_positions": (first["location"].nunique() == len(first), ""),
        "within_region_radius": (bool((dist <= max_r + 1e-6).all()), f"max {dist.max():.1f} km from centre"),
    }
