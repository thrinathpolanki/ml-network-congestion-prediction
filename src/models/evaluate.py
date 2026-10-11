"""Shared evaluation: every model (baseline, XGBoost, LSTM) is scored by THIS code.

Models hand over a *prediction frame* (see build_frame). compare_frames() keeps
only the (site, timestamp) rows ALL models predicted, so no model is judged on
easier or different data.

Metrics
  4-class : accuracy, macro-F1, per-class precision/recall/F1, confusion matrix
  binary  : "congested" = Congested or Critical. precision, recall, F1, ROC-AUC,
            PR-AUC (prevalence is ~4 %, so PR-AUC is the informative one)
  hard cases: recall on NEW onsets (not congested now, congested at t+H) vs on
            continuing congestion; persistence gets the latter for free
  operator: event recall (episodes flagged within their first rows) and false
            alert episodes per site-day (alert-fatigue proxy)
  uncertainty: day-block bootstrap CIs (resamples whole days to respect autocorrelation)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.features.labeling import STATE_NAMES  # noqa: E402

logger = logging.getLogger(__name__)
KEYS = ["site_id", "timestamp"]
N = 4


def build_frame(df: pd.DataFrame, y_pred, p_congested, probs: np.ndarray | None = None) -> pd.DataFrame:
    """Standard prediction frame: keys, y_true, y_pred, p_congested, state_now (+ class probs)."""
    f = df[KEYS + ["target_state_id", "state_now"]].rename(columns={"target_state_id": "y_true"}).copy()
    f["y_true"] = f["y_true"].astype(int)
    f["y_pred"] = np.asarray(y_pred, dtype=int)
    f["p_congested"] = np.asarray(p_congested, dtype=float)
    if probs is not None:
        for k in range(N):
            f[f"p_class{k}"] = probs[:, k]
    return f.reset_index(drop=True)


def _confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.bincount(y.astype(int) * N + p.astype(int), minlength=N * N).reshape(N, N)


def _div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _core(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    sup, pred = cm.sum(1), cm.sum(0)
    prec = np.divide(tp, pred, out=np.zeros(N), where=pred > 0)
    rec = np.divide(tp, sup, out=np.zeros(N), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(N), where=(prec + rec) > 0)
    tpb, fnb = cm[2:, 2:].sum(), cm[2:, :2].sum()
    fpb, tnb = cm[:2, 2:].sum(), cm[:2, :2].sum()
    cp, cr = _div(tpb, tpb + fpb), _div(tpb, tpb + fnb)
    return {
        "accuracy": _div(tp.sum(), cm.sum()), "macro_f1": float(f1.mean()),
        "per_class": {STATE_NAMES[k]: {"precision": float(prec[k]), "recall": float(rec[k]),
                                       "f1": float(f1[k]), "support": int(sup[k])} for k in range(N)},
        "cong_precision": cp, "cong_recall": cr, "cong_f1": _div(2 * cp * cr, cp + cr),
        "cong_false_positive_rate": _div(fpb, fpb + tnb), "recall_critical": float(rec[3]),
    }


def _segments(d: pd.DataFrame, col: str, step: pd.Timedelta) -> pd.Series:
    """Run ids of consecutive equal values per site on a contiguous time grid."""
    gap = d.groupby("site_id")["timestamp"].diff() != step
    new = gap | (d[col] != d[col].shift()) | (d["site_id"] != d["site_id"].shift())
    return new.cumsum()


def event_detection(frame: pd.DataFrame, step: pd.Timedelta, min_len: int, grace: int) -> dict:
    """Episode-level view. Event = run of >= min_len congested TARGET rows."""
    d = frame.sort_values(KEYS).reset_index(drop=True).copy()
    d["yb"], d["pb"] = (d.y_true >= 2).astype(int), (d.y_pred >= 2).astype(int)
    d["run"] = _segments(d, "yb", step)
    d["pos"] = d.groupby("run").cumcount()
    ev = d[d.yb == 1]
    length = ev.groupby("run").size()
    keep = length[length >= min_len].index
    ev = ev[ev.run.isin(keep)]
    if ev.empty:
        return {"n_events": 0, "event_recall": float("nan"), "event_recall_new": float("nan"),
                "n_events_new": 0, "false_alert_episodes_per_site_day": float("nan")}
    detected = ev[ev.pos < grace].groupby("run")["pb"].max()
    first = ev[ev.pos == 0].set_index("run")
    new = first["state_now"] < 2
    det_new = detected.reindex(first.index[new]).fillna(0)
    d["prun"] = _segments(d, "pb", step)
    pred_runs = d[d.pb == 1].groupby("prun")["yb"].max()
    site_days = max(len(d) * step / pd.Timedelta("1D"), 1e-9)
    return {"n_events": int(len(keep)), "event_recall": float(detected.mean()),
            "n_events_new": int(new.sum()),
            "event_recall_new": float(det_new.mean()) if new.any() else float("nan"),
            "false_alert_episodes_per_site_day": float((pred_runs == 0).sum() / site_days)}


def evaluate_frame(frame: pd.DataFrame, cfg: dict) -> dict:
    """Full metric dictionary for one model on one prediction frame."""
    y, p = frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
    cm = _confusion(y, p)
    m = _core(cm)
    yb, pb = y >= 2, p >= 2
    score = frame["p_congested"].to_numpy()
    both = yb.any() and (~yb).any()
    m["roc_auc"] = float(roc_auc_score(yb, score)) if both else float("nan")
    m["pr_auc"] = float(average_precision_score(yb, score)) if yb.any() else float("nan")
    m["prevalence"] = float(yb.mean())
    now = frame["state_now"].to_numpy()
    new, cont = yb & (now < 2), yb & (now >= 2)
    m["n_new_onset"], m["recall_new_onset"] = int(new.sum()), float(pb[new].mean()) if new.any() else float("nan")
    m["n_continuing"], m["recall_continuing"] = int(cont.sum()), float(pb[cont].mean()) if cont.any() else float("nan")
    ec = cfg["evaluation"]
    step = pd.Timedelta(cfg["data"]["sampling_freq"])
    m.update(event_detection(frame, step, int(ec["min_event_steps"]), int(ec["event_grace_steps"])))
    m["n_rows"] = int(len(frame))
    m["confusion_matrix"] = cm.tolist()
    return m


def bootstrap_ci(frame: pd.DataFrame, cfg: dict) -> dict:
    """95 % CIs for macro-F1 and congestion precision/recall/F1 via day-block bootstrap."""
    ec = cfg["evaluation"]
    day = frame["timestamp"].dt.floor("D").to_numpy()
    days = np.unique(day)
    cms = np.stack([_confusion(frame["y_true"].to_numpy()[day == d], frame["y_pred"].to_numpy()[day == d])
                    for d in days])
    rng = np.random.default_rng(int(ec["bootstrap_seed"]))
    draws = [_core(cms[rng.integers(0, len(days), len(days))].sum(0)) for _ in range(int(ec["bootstrap_samples"]))]
    out = {}
    for k in ("macro_f1", "cong_precision", "cong_recall", "cong_f1"):
        v = np.array([d[k] for d in draws])
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def align_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every frame to the (site, timestamp) rows all models predicted."""
    idx = None
    for f in frames.values():
        k = pd.MultiIndex.from_frame(f[KEYS])
        idx = k if idx is None else idx.intersection(k)
    out = {}
    for name, f in frames.items():
        if len(idx) < len(f):
            logger.warning("%s: evaluating on %d of its %d rows (common rows only)", name, len(idx), len(f))
        out[name] = f.set_index(KEYS).loc[idx].reset_index().sort_values(KEYS).reset_index(drop=True)
    return out


def compare_frames(frames: dict[str, pd.DataFrame], cfg: dict, with_ci: bool = True):
    """Evaluate all models on identical rows. Returns (table, results, aligned_frames)."""
    aligned = align_frames(frames)
    results = {n: evaluate_frame(f, cfg) for n, f in aligned.items()}
    rows = []
    for n, r in results.items():
        row = {"model": n, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
               "cong_precision": r["cong_precision"], "cong_recall": r["cong_recall"],
               "cong_f1": r["cong_f1"], "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
               "recall_critical": r["recall_critical"], "recall_new_onset": r["recall_new_onset"],
               "event_recall": r["event_recall"],
               "false_alerts_per_site_day": r["false_alert_episodes_per_site_day"]}
        if with_ci:
            ci = bootstrap_ci(aligned[n], cfg)
            r["ci95"] = ci
            row["macro_f1_ci95"] = f"[{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}]"
            row["cong_recall_ci95"] = f"[{ci['cong_recall'][0]:.3f}, {ci['cong_recall'][1]:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows), results, aligned


def time_predict(fn: Callable[[pd.DataFrame], object], df: pd.DataFrame, repeats: int = 3) -> float:
    """Median wall-clock milliseconds to predict 1,000 rows."""
    sample = df.head(min(len(df), 5000))
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(sample)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) / len(sample) * 1000 * 1000)


def plot_confusions(results: dict, path: str | Path, title: str = "") -> str:
    """Row-normalised confusion matrices side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), squeeze=False)
    for ax, (name, r) in zip(axes[0], results.items()):
        cm = np.array(r["confusion_matrix"], float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(N):
            for j in range(N):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=8)
        ax.set_xticks(range(N)); ax.set_xticklabels([s[:4] for s in STATE_NAMES], fontsize=8)
        ax.set_yticks(range(N)); ax.set_yticklabels([s[:4] for s in STATE_NAMES], fontsize=8)
        ax.set_title(f"{name}\nmacro-F1 {r['macro_f1']:.3f}", fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("true" if ax is axes[0][0] else "")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def calibration_table(y_true_bin: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, dict]:
    """Reliability table (equal-count bins), expected calibration error and Brier score."""
    d = pd.DataFrame({"y": y_true_bin.astype(float), "p": p})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    t = d.groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size"))
    ece = float((t["n"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / t["n"].sum())
    return t.reset_index(), {"ece": ece, "brier": float(((d["p"] - d["y"]) ** 2).mean())}
