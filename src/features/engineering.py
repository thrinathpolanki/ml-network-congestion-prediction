"""Feature engineering shared by TRAINING and INFERENCE.

Every feature at time t is a function of observations at times <= t (plus the
calendar, which is known in advance). Nothing here looks at the future, and
the target (state at t + horizon) is never used as an input.

Two guarantees, both enforced by tests/test_features.py:
  1. CAUSALITY      - changing or removing any row after t cannot change features at t.
  2. WINDOW PARITY  - features computed from only the last `required_history_steps`
                      rows (what the API has) equal those computed from the full
                      history (what training used).

Fitted state (learned from the TRAINING period only): per-site baselines and the
list of regions. Everything else is a deterministic function of the input.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import horizon_steps
from src.data.schema import RAW_METRICS, SchemaError, STATIC_COLUMNS
from src.features.labeling import UNKNOWN_ID, add_derived_indicators, label_dataframe

logger = logging.getLogger(__name__)

KEYS = ["site_id", "timestamp"]
PASSTHROUGH = ["region", "congestion_state_id", "congestion_score", "is_imputed"]
DERIVED_KPIS = ["dl_throughput_per_user_mbps"]
ROLL_STATS = ["mean", "std", "max", "min"]
BASELINE_COLS = ["site_med_latency_ms", "site_p90_util_pct", "site_p90_users"]


class InsufficientHistoryError(ValueError):
    """Raised when too few past observations are available to build features."""


# --------------------------------------------------------------------------- #
# Fitted per-site baselines (non-label statistics, training period only)
# --------------------------------------------------------------------------- #
class SiteBaselines:
    """Typical level of each site: median latency, p90 utilisation, p90 users."""

    def __init__(self) -> None:
        self.table: pd.DataFrame | None = None
        self.region_table: pd.DataFrame | None = None
        self.global_row: pd.Series | None = None
        self.site_region: dict[str, str] = {}
        self._warned: set[str] = set()

    def fit(self, df: pd.DataFrame) -> "SiteBaselines":
        g = df.groupby("site_id")
        self.table = pd.DataFrame({
            "site_med_latency_ms": g["latency_ms"].median(),
            "site_p90_util_pct": g["bandwidth_utilization_pct"].quantile(0.9),
            "site_p90_users": g["active_users"].quantile(0.9),
        })
        site_region = g["region"].first()
        self.site_region = site_region.astype(str).to_dict()
        self.region_table = self.table.groupby(site_region).median()
        self.global_row = self.table.median()
        return self

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the baseline columns for each row (unknown sites fall back to region/global)."""
        if self.table is None:
            raise RuntimeError("SiteBaselines is not fitted.")
        out = pd.DataFrame(index=df.index)
        unknown = sorted(set(df["site_id"].unique()) - set(self.table.index))
        for col in BASELINE_COLS:
            v = df["site_id"].map(self.table[col])
            v = v.fillna(df["region"].map(self.region_table[col])).fillna(self.global_row[col])
            out[col] = v
        new = [s for s in unknown if s not in self._warned]
        if new:
            logger.warning("Unknown sites %s: using region/global baselines.", new)
            self._warned.update(new)
        return out


# --------------------------------------------------------------------------- #
# Feature builder
# --------------------------------------------------------------------------- #
def _time_features(ts: pd.Series, peak_hours: list[int], prefix: str = "", with_hour: bool = False) -> dict:
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    out = {
        f"{prefix}tod_sin": np.sin(2 * np.pi * hour / 24.0),
        f"{prefix}tod_cos": np.cos(2 * np.pi * hour / 24.0),
        f"{prefix}is_weekend": (dow >= 5).astype(float),
        f"{prefix}is_peak_hour": ts.dt.hour.isin(peak_hours).astype(float),
    }
    if with_hour:
        out["hour"] = ts.dt.hour.astype(float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


class FeatureBuilder:
    """Builds the model feature matrix. fit() on training data, then transform() anywhere."""

    def __init__(self, cfg: dict):
        self.cfg = copy.deepcopy(cfg)       # frozen with the artefact -> same labeling at inference
        self.f = self.cfg["features"]
        self.hsteps = horizon_steps(self.cfg)
        self.step = pd.Timedelta(self.cfg["data"]["sampling_freq"])
        self.baselines = SiteBaselines()
        self.regions_: list[str] | None = None

    # ---- column registry (deterministic from config; region columns after fit)
    @property
    def feature_groups(self) -> dict[str, list[str]]:
        f, key = self.f, self.f["key_kpis"]
        g: dict[str, list[str]] = {}
        g["current_kpis"] = RAW_METRICS + DERIVED_KPIS
        g["lags"] = [f"{c}_lag{k}" for c in key for k in f["lags"]]
        g["rate_of_change"] = [f"{c}_diff{k}" for c in key for k in f["roc_steps"]]
        g["rolling"] = [f"{c}_roll{w}_{s}" for c in key for w in f["rolling_windows"] for s in ROLL_STATS]
        g["ewm"] = [f"{c}_ewm{sp}" for c in key for sp in f["ewm_spans"]]
        g["congestion_history"] = (["state_now", "score_now"]
                                   + [f"state_lag{k}" for k in f["state_lags"]]
                                   + [f"score_lag{k}" for k in f["state_lags"]]
                                   + [f"congested_frac_{w}" for w in f["congested_frac_windows"]])
        g["calendar_now"] = ["hour", "tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "is_peak_hour"]
        g["calendar_forecast_time"] = ["fc_tod_sin", "fc_tod_cos", "fc_is_weekend", "fc_is_peak_hour"]
        g["site_static"] = STATIC_COLUMNS + BASELINE_COLS
        g["site_relative"] = ["latency_vs_site", "util_vs_site_p90", "users_vs_site_p90"]
        if self.regions_ is not None:
            g["region_onehot"] = [f"region__{r}" for r in self.regions_]
        return g

    @property
    def feature_columns(self) -> list[str]:
        cols = [c for v in self.feature_groups.values() for c in v]
        if len(cols) != len(set(cols)):
            raise RuntimeError("Duplicate feature names.")
        return cols

    @property
    def required_history_steps(self) -> int:
        """Past rows needed (incl. the current one) for exact feature parity with training."""
        f = self.f
        need = max(max(f["lags"]), max(f["roc_steps"]), max(f["rolling_windows"]) - 1,
                   max(f["state_lags"]), max(f["congested_frac_windows"]) - 1, 6 * max(f["ewm_spans"]))
        return int(need) + 1

    # ---- fitting
    def fit(self, train_df: pd.DataFrame) -> "FeatureBuilder":
        """Learn baselines + region list from TRAINING-period rows only."""
        self._require(train_df)
        self.baselines.fit(train_df)
        self.regions_ = sorted(train_df["region"].dropna().astype(str).unique())
        logger.info("FeatureBuilder fitted on %d rows, %d sites, %d features",
                    len(train_df), train_df["site_id"].nunique(), len(self.feature_columns))
        return self

    @staticmethod
    def _require(df: pd.DataFrame) -> None:
        need = KEYS + ["region"] + RAW_METRICS + STATIC_COLUMNS
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise SchemaError(f"Missing columns for feature building: {missing}")

    # ---- per-site time-series features
    def _site_features(self, g: pd.DataFrame) -> pd.DataFrame:
        f, out = self.f, {}
        for c in RAW_METRICS + DERIVED_KPIS:
            out[c] = g[c]
        for c in f["key_kpis"]:
            s = g[c]
            for k in f["lags"]:
                out[f"{c}_lag{k}"] = s.shift(k)
            for k in f["roc_steps"]:
                out[f"{c}_diff{k}"] = s.diff(k)
            for w in f["rolling_windows"]:
                r = s.rolling(w, min_periods=w)
                out[f"{c}_roll{w}_mean"], out[f"{c}_roll{w}_std"] = r.mean(), r.std()
                out[f"{c}_roll{w}_max"], out[f"{c}_roll{w}_min"] = r.max(), r.min()
            for sp in f["ewm_spans"]:
                out[f"{c}_ewm{sp}"] = s.ewm(span=sp, adjust=False).mean()
        st = g["congestion_state_id"].where(g["congestion_state_id"] != UNKNOWN_ID)
        sc = g["congestion_score"]
        out["state_now"], out["score_now"] = st, sc
        for k in f["state_lags"]:
            out[f"state_lag{k}"], out[f"score_lag{k}"] = st.shift(k), sc.shift(k)
        cong = (st >= 2).astype(float).where(st.notna())
        for w in f["congested_frac_windows"]:
            out[f"congested_frac_{w}"] = cong.rolling(w, min_periods=w).mean()
        return pd.DataFrame(out, index=g.index)

    # ---- main entry point
    def transform(self, df: pd.DataFrame, drop_incomplete: bool = True) -> pd.DataFrame:
        """Build features for every row. Input must be cleaned: regular 5-min grid per site."""
        if self.regions_ is None:
            raise RuntimeError("FeatureBuilder is not fitted. Call fit() on training data first.")
        self._require(df)
        if df.empty:
            raise SchemaError("Empty input.")
        d = df.sort_values(KEYS).reset_index(drop=True)
        gaps = d.groupby("site_id")["timestamp"].diff().dropna()
        if (gaps != self.step).any():
            raise ValueError("Timestamps are not a regular grid with duplicates removed; "
                             "run clean_dataframe() first.")
        d = label_dataframe(add_derived_indicators(d), self.cfg)

        parts = [self._site_features(g) for _, g in d.groupby("site_id", sort=False)]
        feats = pd.concat(parts).loc[d.index]
        base = self.baselines.apply(d)
        feats[STATIC_COLUMNS[0]] = d[STATIC_COLUMNS[0]]
        for c in BASELINE_COLS:
            feats[c] = base[c]
        feats["latency_vs_site"] = d["latency_ms"] / base["site_med_latency_ms"]
        feats["util_vs_site_p90"] = d["bandwidth_utilization_pct"] / base["site_p90_util_pct"]
        feats["users_vs_site_p90"] = d["active_users"] / base["site_p90_users"].clip(lower=1)

        peak = self.f["peak_hours"]
        now = _time_features(d["timestamp"], peak, with_hour=True)
        fc = _time_features(d["timestamp"] + self.step * self.hsteps, peak, prefix="fc_")
        feats = pd.concat([feats, pd.DataFrame(now, index=d.index), pd.DataFrame(fc, index=d.index)], axis=1)
        for r in self.regions_:
            feats[f"region__{r}"] = (d["region"].astype(str) == r).astype(float)

        cols = self.feature_columns
        out = pd.concat([d[KEYS], d[[c for c in PASSTHROUGH if c in d.columns]],
                         feats[cols].astype("float32")], axis=1)
        if drop_incomplete:
            out = out[out[cols].notna().all(axis=1)].reset_index(drop=True)
        return out

    def transform_latest(self, history: pd.DataFrame) -> pd.DataFrame:
        """Features for the most recent row of ONE site's history (used by the API)."""
        if history["site_id"].nunique() != 1:
            raise ValueError("transform_latest expects the history of exactly one site.")
        if len(history) < self.required_history_steps:
            raise InsufficientHistoryError(
                f"Need >= {self.required_history_steps} consecutive readings "
                f"({self.required_history_steps * self.step}); got {len(history)}.")
        tail = history.sort_values("timestamp").tail(self.required_history_steps)
        out = self.transform(tail, drop_incomplete=False).tail(1)
        if out[self.feature_columns].isna().any(axis=None):
            raise InsufficientHistoryError("History contains gaps/missing KPIs in the required window.")
        return out.reset_index(drop=True)

    # ---- persistence & documentation
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "FeatureBuilder":
        obj = joblib.load(path)
        if not isinstance(obj, FeatureBuilder):
            raise TypeError(f"{path} does not contain a FeatureBuilder")
        return obj

    def describe(self) -> str:
        """Markdown feature dictionary with availability + leakage notes."""
        f = self.f
        notes = {
            "current_kpis": ("Raw KPIs reported at time t (+ per-user throughput).", "t", "None: measured at t."),
            "lags": (f"Key KPIs {f['lags']} steps ago.", "t-k", "Past only."),
            "rate_of_change": (f"x[t] - x[t-k], k in {f['roc_steps']}.", "t, t-k", "Past only."),
            "rolling": (f"mean/std/max/min over trailing windows {f['rolling_windows']} (incl. t).", "<= t",
                        "Trailing windows (never centred); strict min_periods."),
            "ewm": (f"Exponentially weighted mean, spans {f['ewm_spans']}.", "<= t", "Recursive filter: causal."),
            "congestion_history": ("Current/previous congestion state & score, share of recent time congested.",
                                   "<= t", "Built from KPIs at <= t via the labeling function; NEVER the target t+H."),
            "calendar_now": ("Hour, time-of-day and day-of-week encodings, weekend, peak-hour flag.", "t", "Calendar: known."),
            "calendar_forecast_time": ("Same calendar flags for the forecast time t+H.", "t (calendar of t+H)",
                                       "Calendar is known in advance; carries no measurement from the future."),
            "site_static": ("Capacity + per-site typical levels (median latency, p90 utilisation, p90 users).",
                            "fitted", "Fitted on TRAINING period only; contain no labels."),
            "site_relative": ("Current KPI divided by the site's own baseline (fixes the region latency confound).",
                              "t + fitted", "As above."),
            "region_onehot": ("One-hot region (categories from training).", "t", "None."),
        }
        lines = ["# Feature dictionary", "",
                 f"Horizon: {self.hsteps} steps ({self.hsteps * self.step}). "
                 f"Sampling step: {self.step}. Required history at inference: "
                 f"{self.required_history_steps} steps.", "",
                 "| Group | # | Description | Uses data from | Leakage control |", "|---|---|---|---|---|"]
        total = 0
        for grp, cols in self.feature_groups.items():
            d, avail, leak = notes[grp]
            lines.append(f"| `{grp}` | {len(cols)} | {d} | {avail} | {leak} |")
            total += len(cols)
        lines += ["", f"**Total features: {total}.**", "",
                  "**Never used as inputs:** `target_state_id`, `target_score`, `target_binary`, "
                  "`timestamp`, `site_id`, `is_imputed` (metadata only), and the cell position columns "
                  "`latitude`, `longitude`, `location` (descriptive metadata for maps; a validation experiment "
                  "found no benefit from adding them, see docs/SPATIAL_FEATURES.md).", ""]
        for grp, cols in self.feature_groups.items():
            lines += [f"### {grp}", ", ".join(f"`{c}`" for c in cols), ""]
        return "\n".join(lines)
