"""Baselines that any ML model must beat to justify its complexity.

All use the same interface:
    fit(train_df) -> self ; predict(df) -> int states 0..3 ;
    congestion_score(df) -> float, higher = more likely congested at t+H (for ROC/PR)

  MajorityClass      always the most common training class
  ThresholdDetector  what an operator rule would do: cut-offs on ONE current KPI
                     (utilisation), fitted on TRAIN to maximise macro-F1
  Persistence        "nothing changes": predicted state at t+H = state now
  SeasonalNaive      "same time yesterday / last week": the state observed at
                     t + H - period (always in the past, so no leakage)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.features.labeling import UNKNOWN_ID  # noqa: F401  (documented sentinel)

KEYS = ["site_id", "timestamp"]
N = 4


def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * N + p, minlength=N * N).reshape(N, N)
    tp, sup, pred = np.diag(cm).astype(float), cm.sum(1), cm.sum(0)
    prec = np.divide(tp, pred, out=np.zeros(N), where=pred > 0)
    rec = np.divide(tp, sup, out=np.zeros(N), where=sup > 0)
    return float(np.divide(2 * prec * rec, prec + rec, out=np.zeros(N), where=(prec + rec) > 0).mean())


class BaselineModel(ABC):
    name = "baseline"

    def fit(self, train: pd.DataFrame) -> "BaselineModel":
        return self

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def congestion_score(self, df: pd.DataFrame) -> np.ndarray: ...


class MajorityClass(BaselineModel):
    name = "Majority class"

    def fit(self, train):
        self.cls_ = int(train["target_state_id"].value_counts().idxmax())
        return self

    def predict(self, df):
        return np.full(len(df), self.cls_, dtype=int)

    def congestion_score(self, df):
        return np.zeros(len(df))


class ThresholdDetector(BaselineModel):
    """Three cut-offs on one current KPI -> NORMAL/WARNING/CONGESTED/CRITICAL (fitted on train)."""

    def __init__(self, feature: str = "bandwidth_utilization_pct"):
        self.feature = feature
        self.name = f"Threshold rule ({feature.split('_')[0]})"

    def fit(self, train):
        x = train[self.feature].to_numpy(float)
        y = train["target_state_id"].to_numpy(int)
        cuts = np.quantile(x, [0.85, 0.95, 0.985])
        grid = np.unique(np.quantile(x, np.linspace(0.3, 0.999, 200)))
        for _ in range(3):                      # coordinate ascent on macro-F1
            for i in range(3):
                lo = cuts[i - 1] if i > 0 else -np.inf
                hi = cuts[i + 1] if i < 2 else np.inf
                best, best_v = _macro_f1(y, np.digitize(x, cuts)), cuts[i]
                for v in grid[(grid > lo) & (grid < hi)]:
                    c = cuts.copy(); c[i] = v
                    s = _macro_f1(y, np.digitize(x, c))
                    if s > best:
                        best, best_v = s, v
                cuts[i] = best_v
        self.cuts_ = cuts
        self.median_ = float(np.nanmedian(x))
        return self

    def _x(self, df):
        return df[self.feature].fillna(self.median_).to_numpy(float)

    def predict(self, df):
        return np.digitize(self._x(df), self.cuts_).astype(int)

    def congestion_score(self, df):
        return self._x(df)


class Persistence(BaselineModel):
    name = "Persistence (state now)"

    def predict(self, df):
        return df["state_now"].fillna(0).to_numpy().astype(int)

    def congestion_score(self, df):
        return df["score_now"].fillna(0).to_numpy(float)


class SeasonalNaive(BaselineModel):
    """State at (t + horizon - period) of the same site; falls back to persistence if unknown."""

    def __init__(self, days: int, horizon_steps: int, step: pd.Timedelta):
        self.period = pd.Timedelta(days=days)
        self.shift = step * horizon_steps - self.period          # negative -> looks into the past
        if self.shift >= pd.Timedelta(0):
            raise ValueError("Seasonal period must be longer than the horizon (otherwise it peeks ahead).")
        self.name = f"Seasonal naive ({days}d)"
        self._state = self._score = None

    def set_history(self, hist: pd.DataFrame) -> "SeasonalNaive":
        """hist = rows with state_now/score_now (train+val+test). Only PAST keys are ever read."""
        h = hist.drop_duplicates(KEYS).set_index(KEYS)
        self._state, self._score = h["state_now"], h["score_now"]
        return self

    def _lookup(self, series: pd.Series, df: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
        if series is None:
            raise RuntimeError("Call set_history() before predicting.")
        keys = pd.MultiIndex.from_arrays([df["site_id"], df["timestamp"] + self.shift])
        v = series.reindex(keys).to_numpy(float)
        return np.where(np.isnan(v), fallback, v)

    def predict(self, df):
        return self._lookup(self._state, df, df["state_now"].fillna(0).to_numpy(float)).astype(int)

    def congestion_score(self, df):
        return self._lookup(self._score, df, df["score_now"].fillna(0).to_numpy(float))
