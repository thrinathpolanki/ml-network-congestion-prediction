"""Robust cleaning of raw network KPIs.

Order of operations (each step is counted in the returned report):
  1. parse timestamps, coerce metrics to numbers, snap timestamps to the sampling grid
  2. remove duplicate (site, timestamp) rows
  3. invalid (physically impossible) values -> NaN
  4. rebuild a regular time grid per site (gaps become NaN rows)
  5. isolated outliers (sensor glitches) -> NaN   [Hampel filter, offline only]
  6. short gaps (<= max_gap_steps) linearly interpolated; long gaps stay NaN

`sanitize_metrics` (steps 1 + 3) is the part that is also safe for single
real-time readings; the API reuses it so training and serving agree.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.schema import (LOCATION_COLUMNS, PROVENANCE_COLUMN, RAW_METRICS, STATIC_COLUMNS,
                             VALID_RANGES, validate_schema)

logger = logging.getLogger(__name__)
_COUNT_COLUMNS = ["active_users", "active_sessions"]


def sanitize_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Coerce metrics to numeric and set out-of-range values to NaN."""
    out = df.copy()
    nullified: dict[str, int] = {}
    for col in RAW_METRICS + STATIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col, (lo, hi) in VALID_RANGES.items():
        if col in out.columns:
            bad = (out[col] < lo) | (out[col] > hi)
            if bad.any():
                nullified[col] = int(bad.sum())
                out.loc[bad, col] = np.nan
    return out, nullified


def _hampel_isolated(s: pd.Series, window: int, k: float, rel_floor: float) -> pd.Series:
    """Flag single-step deviations from the local median (isolated glitches).

    Genuine congestion episodes last several steps, so neighbouring points also
    deviate and are NOT flagged; a one-sample collector glitch is.
    """
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    scale = np.maximum(1.4826 * mad, rel_floor * med.abs()).clip(lower=1e-6)
    flag = (s - med).abs() > k * scale
    prev_flag = flag.shift(1, fill_value=False)
    next_flag = flag.shift(-1, fill_value=False)
    return flag & ~prev_flag & ~next_flag


def _interp_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """Linear interpolation only inside NaN runs no longer than max_gap."""
    isna = s.isna()
    run_len = isna.groupby((~isna).cumsum()).transform("sum")
    filled = s.interpolate(method="linear", limit_area="inside")
    filled[isna & (run_len > max_gap)] = np.nan
    return filled


def _process_site(g: pd.DataFrame, site: str, freq: str, pcfg: dict, stats: dict) -> pd.DataFrame:
    idx = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq=freq)
    g = g.set_index("timestamp").reindex(idx)
    g.index.name = "timestamp"
    g["site_id"] = site
    for c in ["region", PROVENANCE_COLUMN] + STATIC_COLUMNS + LOCATION_COLUMNS:
        if c in g.columns:
            g[c] = g[c].ffill().bfill()

    oc = pcfg["outlier"]
    if oc.get("enabled", True):
        for col, p in oc["columns"].items():
            if col in g.columns:
                flagged = _hampel_isolated(g[col], int(oc["window"]),
                                           float(p["n_sigmas"]), float(p["rel_floor"]))
                if flagged.any():
                    stats["outliers_removed"][col] = stats["outliers_removed"].get(col, 0) + int(flagged.sum())
                    g.loc[flagged, col] = np.nan

    before_na = g[RAW_METRICS].isna()
    max_gap = int(pcfg["max_gap_steps"])
    for col in RAW_METRICS:
        g[col] = _interp_short_gaps(g[col], max_gap)
    for col in _COUNT_COLUMNS:
        g[col] = g[col].round()
    after_na = g[RAW_METRICS].isna()
    filled = before_na & ~after_na
    g["is_imputed"] = filled.any(axis=1)
    stats["cells_imputed"] += int(filled.to_numpy().sum())
    return g.reset_index()


def clean_dataframe(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Clean raw KPI data. Returns (clean_df, report)."""
    validate_schema(df)
    pcfg, freq = cfg["preprocessing"], cfg["data"]["sampling_freq"]
    report: dict = {"rows_in": int(len(df))}
    out = df.copy()

    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    out["timestamp"] = ts.dt.round(freq)
    bad_rows = out["timestamp"].isna() | out["site_id"].isna()
    report["rows_dropped_bad_timestamp_or_site"] = int(bad_rows.sum())
    out = out.loc[~bad_rows]

    out, nullified = sanitize_metrics(out)
    report["invalid_values_nullified"] = nullified

    dup = out.duplicated(["site_id", "timestamp"], keep="first")
    report["duplicates_removed"] = int(dup.sum())
    out = out.loc[~dup].sort_values(["site_id", "timestamp"])
    rows_before_grid = len(out)

    stats = {"rows_inserted_for_gaps": 0, "outliers_removed": {}, "cells_imputed": 0}
    parts = [_process_site(g, str(site), freq, pcfg, stats)
             for site, g in out.groupby("site_id", sort=True)]
    clean = pd.concat(parts, ignore_index=True)

    report["rows_inserted_for_gaps"] = int(len(clean) - rows_before_grid)
    report["outliers_removed"] = stats["outliers_removed"]
    report["cells_imputed"] = stats["cells_imputed"]
    report["rows_with_unfilled_gaps"] = int(clean[RAW_METRICS].isna().any(axis=1).sum())
    report["rows_out"] = int(len(clean))
    report["sites"] = int(clean["site_id"].nunique())
    logger.info("Cleaning report: %s", report)
    return clean, report
