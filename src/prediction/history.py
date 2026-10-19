"""Online counterpart of the offline cleaning step, for readings that arrive from an API.

Offline training data was cleaned with future-aware tools (linear interpolation, a centred
Hampel filter). A live system only knows the past, so here we do the causal subset:
  coerce types, snap timestamps to the grid, drop duplicates (keep the newest), set impossible
  values to NaN, rebuild a regular grid, forward-fill gaps of at most `max_gap_steps`.
Known difference: isolated one-sample sensor glitches are NOT removed online (that needs the
future); the alert engine's debounce (Phase 12) is the mitigation.
"""
from __future__ import annotations

import pandas as pd

from src.data.schema import LOCATION_COLUMNS, PROVENANCE_COLUMN, RAW_METRICS, STATIC_COLUMNS, validate_schema
from src.preprocessing.cleaning import sanitize_metrics


def _ffill_short(s: pd.Series, max_gap: int) -> pd.Series:
    """Forward-fill NaN runs no longer than max_gap; longer runs stay NaN entirely."""
    isna = s.isna()
    run_len = isna.groupby((~isna).cumsum()).transform("sum")
    out = s.ffill()
    out[isna & (run_len > max_gap)] = float("nan")
    return out


def prepare_history(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Raw readings (any number of sites, any order) -> regular 5-min grid per site, ready for FeatureBuilder."""
    validate_schema(df)
    freq, max_gap = cfg["data"]["sampling_freq"], int(cfg["preprocessing"]["max_gap_steps"])
    d = df.copy()
    ts = pd.to_datetime(d["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    d["timestamp"] = ts.dt.round(freq)
    d = d.loc[d["timestamp"].notna() & d["site_id"].notna()]
    d, _ = sanitize_metrics(d)
    d = d.sort_values(["site_id", "timestamp"], kind="stable").drop_duplicates(["site_id", "timestamp"], keep="last")
    parts = []
    for site, g in d.groupby("site_id", sort=True):
        idx = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq=freq)
        g = g.set_index("timestamp").reindex(idx)
        g.index.name = "timestamp"
        g["site_id"] = site
        for c in ["region", PROVENANCE_COLUMN] + STATIC_COLUMNS + LOCATION_COLUMNS:
            if c in g.columns:
                g[c] = g[c].ffill().bfill()
        for c in RAW_METRICS:
            g[c] = _ffill_short(g[c], max_gap)
        parts.append(g.reset_index())
    if not parts:
        raise ValueError("No usable rows after cleaning the history.")
    return pd.concat(parts, ignore_index=True)
