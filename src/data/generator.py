"""Synthetic telecom traffic generator.

!!! SIMULATED DATA !!!  Nothing produced here is a real telecom measurement.
It is built to have realistic *structure* (daily/weekly cycles, temporal
autocorrelation, load-dependent degradation, regional differences, spikes,
sustained episodes, sensor glitches) so the ML pipeline can be developed and
tested. Results on this data must NOT be presented as production performance.

Causal story per cell:
    offered load d(t) = peak_load * daily_profile * day_level * AR(1) noise * events
    utilisation       = soft-saturating function of d(t)
    latency / loss / jitter / drops / retransmissions grow non-linearly as
    utilisation approaches saturation; per-user throughput falls.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from src.data.schema import LOCATION_COLUMNS, PROVENANCE_COLUMN, RAW_METRICS

logger = logging.getLogger(__name__)

SYNTHETIC_WARNING = (
    "SIMULATED DATA. Not real telecom measurements; do not report results "
    "obtained on it as production network performance."
)


# --------------------------------------------------------------------------- #
# Demand building blocks
# --------------------------------------------------------------------------- #
def _bump(hour: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian bump on a 24-hour circle."""
    d = np.abs(hour - mu)
    d = np.minimum(d, 24.0 - d)
    return np.exp(-0.5 * (d / sigma) ** 2)


def _raw_profile(hour: np.ndarray, weekend: np.ndarray, kind: str) -> np.ndarray:
    resid_wd = 0.12 + 0.45 * _bump(hour, 8.5, 1.3) + 0.35 * _bump(hour, 13, 2.0) + 1.0 * _bump(hour, 20.5, 2.0)
    resid_we = 0.15 + 0.35 * _bump(hour, 10.5, 2.0) + 0.40 * _bump(hour, 14, 2.5) + 0.95 * _bump(hour, 21, 2.3)
    bus_wd = 0.08 + 0.90 * _bump(hour, 10.5, 1.6) + 0.85 * _bump(hour, 15, 1.8) + 0.30 * _bump(hour, 20, 2.0)
    bus_we = 0.08 + 0.25 * _bump(hour, 13, 3.0)
    if kind == "residential":
        wd, we = resid_wd, resid_we
    elif kind == "business":
        wd, we = bus_wd, bus_we
    elif kind == "mixed":
        wd, we = 0.5 * (resid_wd + bus_wd), 0.5 * (resid_we + bus_we)
    else:
        raise ValueError(f"Unknown traffic profile '{kind}'")
    return np.where(weekend, we, wd)


def daily_profile(hour: np.ndarray, weekend: np.ndarray, kind: str) -> np.ndarray:
    """Normalised (max ~ 1 on weekdays) demand shape for a traffic profile."""
    grid = np.linspace(0, 24, 577)
    norm = _raw_profile(grid, np.zeros_like(grid, dtype=bool), kind).max()
    return _raw_profile(hour, weekend, kind) / norm


def _ar1_lognormal(n: int, rng: np.random.Generator, phi: float, sigma: float) -> np.ndarray:
    """Mean-one log-normal factor with AR(1) log-noise of stationary std `sigma`."""
    eps = rng.normal(0.0, sigma * np.sqrt(1 - phi**2), size=n)
    x = lfilter([1.0], [1.0, -phi], eps)
    return np.exp(x - 0.5 * sigma**2)


def _event_factor(
    n: int, step_min: float, days: float, rng: np.random.Generator,
    rate_per_day: float, dur_minutes: list, magnitude: list, kind: str,
) -> np.ndarray:
    """Multiplicative load factor from random events (spikes or sustained)."""
    factor = np.ones(n)
    n_events = rng.poisson(max(rate_per_day * days, 0.0))
    for _ in range(n_events):
        dur = max(int(rng.uniform(*dur_minutes) / step_min), 2)
        start = int(rng.integers(0, n))
        mag = rng.uniform(*magnitude)
        if kind == "spike":      # fast rise, linear decay
            shape = np.interp(np.linspace(0, 1, dur), [0, 0.2, 1.0], [0, 1, 0])
        else:                    # sustained: ramp up, plateau, ramp down
            shape = np.interp(np.linspace(0, 1, dur), [0, 0.15, 0.85, 1.0], [0, 1, 1, 0])
        end = min(start + dur, n)
        seg = 1.0 + (mag - 1.0) * shape[: end - start]
        factor[start:end] = np.maximum(factor[start:end], seg)
    return factor


# --------------------------------------------------------------------------- #
# Cell positions
# --------------------------------------------------------------------------- #
def site_locations(loc_cfg: dict, regions_of_sites: list[str], seed: int) -> list[tuple[float, float]]:
    """One fixed (lat, lon) per cell, scattered around a centre; dense regions sit closer to it.

    Uses its OWN seed sequence, so adding or changing coordinates never alters any simulated KPI value.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7919]))
    lat0, lon0 = float(loc_cfg["center_lat"]), float(loc_cfg["center_lon"])
    out = []
    for region in regions_of_sites:
        radius = float(loc_cfg["radius_km"][region])
        dist = radius * np.sqrt(rng.uniform(0.02, 1.0))            # uniform over the disc (avoids piling up in the middle)
        bearing = rng.uniform(0, 2 * np.pi)
        dlat = dist * np.cos(bearing) / 111.32
        dlon = dist * np.sin(bearing) / (111.32 * np.cos(np.radians(lat0)))
        out.append((round(lat0 + dlat, 6), round(lon0 + dlon, 6)))
    return out


# --------------------------------------------------------------------------- #
# One site
# --------------------------------------------------------------------------- #
def _simulate_site(
    site_id: str, region: str, rp: dict, ts: pd.DatetimeIndex, gcfg: dict,
    step_s: float, rng: np.random.Generator, position: tuple[float, float],
) -> pd.DataFrame:
    n = len(ts)
    step_min = step_s / 60.0
    days = n * step_min / 1440.0
    hour = np.asarray(ts.hour, dtype=float) + np.asarray(ts.minute, dtype=float) / 60.0
    weekend = np.asarray(ts.dayofweek) >= 5
    intensity = float(gcfg["congestion_intensity"])
    nz, ev = gcfg["noise"], gcfg["events"]

    # --- site personality -------------------------------------------------
    capacity = rp["capacity_mbps"] * rng.uniform(0.9, 1.1)
    peak_load = rp["peak_load"] * rng.lognormal(0.0, 0.08)
    users_cap = rp["users_at_capacity"] * rng.uniform(0.9, 1.1)
    sess_per_user = rng.uniform(1.4, 2.2)
    base_lat = rp["base_latency_ms"] * rng.uniform(0.85, 1.2)
    sinr0 = rp["sinr_db"] + rng.normal(0, 1.0)

    # --- offered load ratio d(t) ------------------------------------------
    phi = float(np.exp(-step_min / float(nz["autocorr_minutes"])))
    day_idx = np.asarray((ts - ts[0]).days, dtype=int)
    day_level = np.exp(rng.normal(0, nz["daily_sigma"], size=day_idx.max() + 1))[day_idx]
    prof = daily_profile(hour, weekend, rp["profile"])
    noise = _ar1_lognormal(n, rng, phi, nz["demand_sigma"])
    spikes = _event_factor(n, step_min, days, rng, ev["spike_rate_per_day"] * intensity,
                           ev["spike_duration_minutes"], ev["spike_magnitude"], "spike")
    sustained = _event_factor(n, step_min, days, rng, ev["sustained_rate_per_day"] * intensity,
                              ev["sustained_duration_minutes"], ev["sustained_magnitude"], "sustained")
    d = peak_load * intensity * prof * day_level * noise * np.maximum(spikes, sustained)

    # --- load -> utilisation (soft saturation) ----------------------------
    u = d / (1.0 + d**4) ** 0.25                     # 0..1
    u = np.clip(u + rng.normal(0, 0.008, n), 0.003, 1.0)
    p = np.clip((d - 0.80) / 0.50, 0.0, 1.5)         # congestion pressure
    uq = np.minimum(u, 0.985)

    def ln(s: float) -> np.ndarray:
        return rng.lognormal(0.0, s, n)

    latency = (base_lat + 9.0 * uq / (1.02 - uq) + 25.0 * p) * ln(0.08)
    packet_loss = (0.03 + 0.15 * u**2 + 2.6 * p**1.4) * ln(0.25)
    jitter = np.maximum(
    1.2 + 0.12 * (latency - base_lat) + 0.5 * p,
    0.0
) * ln(0.15)
    dropped = (0.10 + 0.35 * u**2 + 2.6 * p**1.5) * ln(0.20)
    handover = (0.30 + 0.50 * u**2 + 2.0 * p) * ln(0.25)
    retrans = (0.60 + 1.50 * u**2 + 4.5 * p) * ln(0.20)
    cpu = np.clip(12 + 75 * u**1.3 + rng.normal(0, 2.5, n), 1, 100)
    sinr = sinr0 - 4.0 * np.minimum(p, 1.0) + rng.normal(0, 1.2, n)

    total_mbps = capacity * u
    squeeze = 1.0 - 0.20 * np.minimum(p, 1.0)        # efficiency loss under congestion
    dl = total_mbps * 0.85 * squeeze * ln(0.04)
    ul = total_mbps * 0.15 * squeeze * ln(0.06)
    users = rng.poisson(np.clip(d, 0, 1.6) * users_cap).astype(float)
    sessions = np.round(users * sess_per_user * ln(0.05))
    volume = (dl + ul) * step_s / 8000.0             # Mbit -> GB

    return pd.DataFrame({
        "timestamp": ts, "site_id": site_id, "region": region,
        LOCATION_COLUMNS[0]: position[0], LOCATION_COLUMNS[1]: position[1],
        LOCATION_COLUMNS[2]: f"{position[0]:.6f},{position[1]:.6f}",
        "site_capacity_mbps": round(capacity, 1),
        "active_users": users, "active_sessions": sessions, "traffic_volume_gb": volume,
        "bandwidth_utilization_pct": 100 * u,
        "download_throughput_mbps": dl, "upload_throughput_mbps": ul,
        "latency_ms": latency, "packet_loss_pct": packet_loss, "jitter_ms": jitter,
        "signal_quality_sinr_db": sinr, "cpu_utilization_pct": cpu,
        "dropped_call_rate_pct": dropped, "handover_failure_rate_pct": handover,
        "retransmission_rate_pct": retrans,
    })


# --------------------------------------------------------------------------- #
# Data-quality problems (so the cleaning phase has something real to do)
# --------------------------------------------------------------------------- #
def _inject_quality_issues(df: pd.DataFrame, q: dict, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # 1. sensor/collector glitches (extreme but "plausible-looking" values)
    idx = np.flatnonzero(rng.random(n) < q["anomaly_rate"])
    for i in idx:
        kind = rng.integers(0, 3)
        if kind == 0:
            df.iat[i, df.columns.get_loc("latency_ms")] *= rng.uniform(4, 8)
        elif kind == 1:
            df.iat[i, df.columns.get_loc("packet_loss_pct")] += rng.uniform(5, 15)
        else:
            for c in ("download_throughput_mbps", "upload_throughput_mbps"):
                df.iat[i, df.columns.get_loc(c)] *= 0.02  # collector drop-out

    # 2. impossible values
    for i in np.flatnonzero(rng.random(n) < q["invalid_value_rate"]):
        c = rng.choice(["download_throughput_mbps", "bandwidth_utilization_pct", "latency_ms"])
        df.iat[i, df.columns.get_loc(c)] = {"download_throughput_mbps": -1.0,
                                            "bandwidth_utilization_pct": 140.0,
                                            "latency_ms": -5.0}[c]

    # 3. missing cells
    for col in RAW_METRICS:
        df.loc[rng.random(n) < q["missing_value_rate"], col] = np.nan

    # 4. missing timestamps (rows dropped -> gaps)
    df = df.loc[rng.random(n) >= q["missing_row_rate"]]

    # 5. duplicate rows
    dups = df.loc[rng.random(len(df)) < q["duplicate_row_rate"]]
    df = pd.concat([df, dups]).sort_values(["site_id", "timestamp"], kind="stable")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_dataset(
    cfg: dict, *, n_sites: int | None = None, start: str | None = None,
    end: str | None = None, freq: str | None = None, seed: int | None = None,
    intensity: float | None = None, inject_issues: bool | None = None,
) -> pd.DataFrame:
    """Generate a synthetic multi-site dataset. Keyword args override config.

    Identical arguments + seed always give an identical DataFrame.
    """
    g = json.loads(json.dumps(cfg["generator"]))  # deep copy, config untouched
    if intensity is not None:
        g["congestion_intensity"] = intensity
    if inject_issues is not None:
        g["data_quality"]["inject_issues"] = inject_issues
    n_sites = int(g["n_sites"] if n_sites is None else n_sites)
    start = g["start"] if start is None else start
    end = g["end"] if end is None else end
    freq = cfg["data"]["sampling_freq"] if freq is None else freq
    seed = int(cfg["project"]["seed"] if seed is None else seed)
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    if g["congestion_intensity"] <= 0:
        raise ValueError("congestion_intensity must be > 0")

    ts = pd.date_range(start, end, freq=freq, inclusive="left")
    if len(ts) < 2:
        raise ValueError("Date range / frequency produce fewer than 2 samples.")
    step_s = pd.Timedelta(freq).total_seconds()

    master = np.random.SeedSequence(seed)
    assign_ss, quality_ss, *site_ss = master.spawn(n_sites + 2)
    regions = g["regions"]
    names = list(regions)
    shares = np.array([regions[r]["share"] for r in names], dtype=float)
    assigned = np.random.default_rng(assign_ss).choice(names, size=n_sites, p=shares / shares.sum())

    positions = site_locations(g["location"], [str(r) for r in assigned], seed)
    frames = []
    for i in range(n_sites):
        rname = str(assigned[i])
        frames.append(_simulate_site(f"CELL_{i + 1:03d}", rname, regions[rname], ts, g,
                                     step_s, np.random.default_rng(site_ss[i]), positions[i]))
    df = pd.concat(frames, ignore_index=True)

    if g["data_quality"]["inject_issues"]:
        df = _inject_quality_issues(df, g["data_quality"], np.random.default_rng(quality_ss))

    df[PROVENANCE_COLUMN] = cfg["data"].get("data_source_label", "synthetic")
    logger.info("Generated %d rows | %d sites | %s -> %s | freq=%s | seed=%d | SIMULATED DATA",
                len(df), n_sites, start, end, freq, seed)
    return df


def save_dataset(df: pd.DataFrame, path: str | Path, cfg: dict, seed: int | None = None) -> Path:
    """Write CSV or Parquet (by extension) plus a .meta.json provenance sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Output must end with .csv or .parquet")
    meta = {
        "data_source": cfg["data"].get("data_source_label", "synthetic"),
        "warning": SYNTHETIC_WARNING,
        "rows": int(len(df)),
        "sites": int(df["site_id"].nunique()),
        "time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "seed": seed if seed is not None else cfg["project"]["seed"],
        "generator_config": cfg["generator"],
        "sampling_freq": cfg["data"]["sampling_freq"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Saved dataset to %s", path)
    return path
