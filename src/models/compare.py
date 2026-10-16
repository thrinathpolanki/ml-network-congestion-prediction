"""Logic for the formal model comparison (Phase 10).

Process rules enforced here and in scripts/compare_models.py
  * models are selected on VALIDATION evidence only, using a rule written down in advance
  * the test set is scored once (a lock file prevents accidental re-scoring)
  * seed-to-seed variance is measured and used as the yardstick for "is it really better?"
  * a hybrid is evaluated on test only if validation justified adopting it
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.models.evaluate import build_frame, evaluate_frame
from src.models.xgb_model import select_threshold

KEYS = ["site_id", "timestamp"]


# --------------------------------------------------------------------------- #
# Common operating-point policy (calibration + F-beta threshold) for ANY probability model
# --------------------------------------------------------------------------- #
@dataclass
class OperatingPoint:
    cal_x: np.ndarray | None
    cal_y: np.ndarray | None
    tau: float

    @staticmethod
    def fit(p_val: np.ndarray, y_val: np.ndarray, thr_cfg: dict, calibrate: bool = True) -> "OperatingPoint":
        raw, yb = p_val[:, 2:].sum(1), (y_val >= 2)
        cx = cy = None
        if calibrate:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, yb.astype(float))
            cx, cy = iso.X_thresholds_.astype(float), iso.y_thresholds_.astype(float)
        op = OperatingPoint(cx, cy, 0.5)
        grid = np.linspace(thr_cfg["grid_min"], thr_cfg["grid_max"], int(thr_cfg["grid_points"]))
        op.tau, _ = select_threshold(yb, op.prob(p_val), float(thr_cfg["beta"]), grid)
        return op

    def prob(self, p: np.ndarray) -> np.ndarray:
        raw = p[:, 2:].sum(1)
        return raw if self.cal_x is None else np.interp(raw, self.cal_x, self.cal_y)

    def decide(self, p: np.ndarray, tau: float | None = None) -> np.ndarray:
        t = self.tau if tau is None else tau
        return np.where(self.prob(p) >= t, 2 + p[:, 2:].argmax(1), p[:, :2].argmax(1))


# --------------------------------------------------------------------------- #
# Operator curve and its summary
# --------------------------------------------------------------------------- #
def operator_curve(op: OperatingPoint, p: np.ndarray, meta: pd.DataFrame, cfg: dict, taus: list[float]) -> pd.DataFrame:
    """Sweep tau: row-level precision/recall plus event-level recall and false alert episodes per site-day."""
    rows = []
    for t in taus:
        r = evaluate_frame(build_frame(meta, op.decide(p, t), p[:, 2:].sum(1)), cfg)
        rows.append({"tau": t, "precision": r["cong_precision"], "recall": r["cong_recall"],
                     "new_onset_recall": r["recall_new_onset"], "event_recall": r["event_recall"],
                     "false_alerts_per_site_day": r["false_alert_episodes_per_site_day"]})
    return pd.DataFrame(rows)


def operator_area(curve: pd.DataFrame, fa_range: tuple[float, float]) -> float:
    """Mean event recall over a range of false-alert rates (curve interpolated; ends held flat)."""
    c = curve.dropna(subset=["event_recall"]).sort_values("false_alerts_per_site_day")
    grid = np.linspace(fa_range[0], fa_range[1], 41)
    return float(np.interp(grid, c["false_alerts_per_site_day"].to_numpy(), c["event_recall"].to_numpy()).mean())


# --------------------------------------------------------------------------- #
# Selection rule (fixed in advance, validation evidence only)
# --------------------------------------------------------------------------- #
def pooled_z(base: list[float], chal: list[float], sd_floor: float = 1e-3) -> float:
    """(mean(chal) - mean(base)) / pooled SD across seeds. Crude with few seeds: a yardstick, not a p-value."""
    b, c = np.asarray(base, float), np.asarray(chal, float)
    sd = np.sqrt((b.var(ddof=1) + c.var(ddof=1)) / 2) if len(b) > 1 and len(c) > 1 else 0.0
    return float((c.mean() - b.mean()) / max(sd, sd_floor))


def select_model(val: dict[str, dict[str, list[float]]], default: str, z_min: float) -> dict:
    """Keep `default` unless a challenger is clearly better on >=1 criterion and not clearly worse on the other.

    criteria (both on validation, per seed): 'pr_auc' (threshold-free ranking quality) and
    'operator_area' (event recall averaged over a false-alert budget).
    """
    decision = {"default": default, "z_min": z_min, "comparisons": {}, "chosen": default, "reason": ""}
    n_seeds = min(len(v) for m in val.values() for v in m.values())
    if n_seeds < 3:
        decision["reason"] = f"Only {n_seeds} seed(s): variance cannot be estimated, so the default ({default}) is kept."
        return decision
    wins = []
    for name, metrics in val.items():
        if name == default:
            continue
        z = {k: pooled_z(val[default][k], metrics[k]) for k in ("pr_auc", "operator_area")}
        decision["comparisons"][name] = z
        if max(z.values()) >= z_min and min(z.values()) > -z_min:
            wins.append((max(z.values()), name))
    if wins:
        decision["chosen"] = max(wins)[1]
        decision["reason"] = (f"{decision['chosen']} beats {default} by >= {z_min} pooled seed-SDs on at least one "
                              f"validation criterion without being clearly worse on the other.")
    else:
        decision["reason"] = (f"No challenger beat {default} by >= {z_min} pooled seed-SDs on validation "
                              f"(without being clearly worse elsewhere); keeping the simplest robust option.")
    return decision


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def error_overlap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """How much do two models' mistakes coincide? (binary congested decision)"""
    y, a, b = y_true >= 2, pred_a >= 2, pred_b >= 2
    miss_a, miss_b = y & ~a, y & ~b
    fa_a, fa_b = ~y & a, ~y & b
    jac = lambda u, v: float((u & v).sum() / max((u | v).sum(), 1))  # noqa: E731
    return {"n_congested": int(y.sum()), "missed_by_a": int(miss_a.sum()), "missed_by_b": int(miss_b.sum()),
            "missed_by_both": int((miss_a & miss_b).sum()), "miss_jaccard": jac(miss_a, miss_b),
            "false_alarm_a": int(fa_a.sum()), "false_alarm_b": int(fa_b.sum()),
            "false_alarm_both": int((fa_a & fa_b).sum()), "false_alarm_jaccard": jac(fa_a, fa_b),
            "caught_by_either": int((y & (a | b)).sum()), "caught_by_a": int((y & a).sum()), "caught_by_b": int((y & b).sum())}


def time_requests(fn: Callable[[int], object], n: int, warmup: int = 5) -> dict:
    """Wall-clock latency (ms) of fn(i) for i in range(n), after warm-up."""
    for i in range(warmup):
        fn(i)
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "n": n}


def align_probs(meta_a: pd.DataFrame, p_a: np.ndarray, meta_b: pd.DataFrame, p_b: np.ndarray):
    """Restrict two probability sets to their common (site, timestamp) rows, same order."""
    ia, ib = pd.MultiIndex.from_frame(meta_a[KEYS]), pd.MultiIndex.from_frame(meta_b[KEYS])
    common = ia.intersection(ib)
    pa, pb = ia.get_indexer(common), ib.get_indexer(common)
    return meta_a.iloc[pa].reset_index(drop=True), p_a[pa], p_b[pb]


def summarize(values: list[float]) -> dict:
    v = np.asarray(values, float)
    return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
            "min": float(np.nanmin(v)), "max": float(np.nanmax(v)), "n": int(len(v))}
