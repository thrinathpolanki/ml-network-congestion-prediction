"""Data contract between ANY data source and the ML system.

A real telecom adapter only has to produce a DataFrame that satisfies this
schema (see src/data/sources.py). Everything downstream relies on it.
"""
from __future__ import annotations

import pandas as pd

# --- identifiers ------------------------------------------------------------
ID_COLUMNS = ["timestamp", "site_id", "region"]
PROVENANCE_COLUMN = "data_source"          # 'synthetic' | 'real' | name of feed
STATIC_COLUMNS = ["site_capacity_mbps"]    # comes from network inventory
# Optional descriptive metadata (cell position). Carried through the pipeline for maps/dashboards but
# NOT required and NOT used as model input. "location" is the combined "lat,lon" string.
LOCATION_COLUMNS = ["latitude", "longitude", "location"]

# --- RAW MEASUREMENTS (what a network monitoring system reports) ------------
RAW_METRICS = [
    "active_users",
    "active_sessions",
    "traffic_volume_gb",
    "bandwidth_utilization_pct",
    "download_throughput_mbps",
    "upload_throughput_mbps",
    "latency_ms",
    "packet_loss_pct",
    "jitter_ms",
    "signal_quality_sinr_db",
    "cpu_utilization_pct",
    "dropped_call_rate_pct",
    "handover_failure_rate_pct",
    "retransmission_rate_pct",
]

REQUIRED_COLUMNS = ID_COLUMNS + STATIC_COLUMNS + RAW_METRICS

# --- ENGINEERED FEATURES (derived later, Phase 6) ---------------------------
# hour, day_of_week, is_weekend, is_peak_hour, lag_*, rolling_*, ewm_*, ...
# --- TARGETS (derived in src/features/labeling.py) --------------------------
# congestion_score, congestion_state, congestion_state_id (current state)
# target_state_id, target_binary, target_score (state at t + horizon)

# Physically plausible ranges. Values outside are treated as INVALID (not just
# outliers) by the cleaning step. SINR may legitimately be negative.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "active_users": (0, 100_000),
    "active_sessions": (0, 500_000),
    "traffic_volume_gb": (0, 10_000),
    "bandwidth_utilization_pct": (0, 100),
    "download_throughput_mbps": (0, 20_000),
    "upload_throughput_mbps": (0, 10_000),
    "latency_ms": (0, 10_000),
    "packet_loss_pct": (0, 100),
    "jitter_ms": (0, 5_000),
    "signal_quality_sinr_db": (-30, 50),
    "cpu_utilization_pct": (0, 100),
    "dropped_call_rate_pct": (0, 100),
    "handover_failure_rate_pct": (0, 100),
    "retransmission_rate_pct": (0, 100),
}


class SchemaError(ValueError):
    """Raised when a DataFrame does not satisfy the data contract."""


def validate_schema(df: pd.DataFrame, required: list[str] | None = None) -> None:
    """Raise SchemaError if required columns are missing."""
    required = required or REQUIRED_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")
    if df.empty:
        raise SchemaError("DataFrame is empty.")


def count_invalid_values(df: pd.DataFrame) -> dict[str, int]:
    """Count out-of-range values per metric (NaN is not counted as invalid)."""
    out: dict[str, int] = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            n = int(((s < lo) | (s > hi)).sum())
            if n:
                out[col] = n
    return out
