"""Risk engine: model probabilities -> operator-facing congestion risk assessment.

Everything operational is configuration (config['risk']): the probability cut-offs of the
risk levels and the recommended-action texts. The prediction horizon comes from
config['prediction'] and is validated against the trained model (a different horizon
needs a retrained model, not just a different number).

Definitions
  congestion_probability  calibrated P(CONGESTED or CRITICAL at t + horizon)
  risk_level              LOW / MEDIUM / HIGH / CRITICAL from that probability
  predicted_state         NORMAL / WARNING / CONGESTED / CRITICAL (model decision at its tuned alert threshold)
  (both are bounded by risk.probability_bounds: never reported as exactly 0 or 1)
  confidence              probability of the predicted BINARY side: P(cong) if the state is
                          CONGESTED/CRITICAL else 1 - P(cong). Calibrated, not a class-argmax score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from src.features.labeling import STATE_NAMES

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def horizon_label(minutes: float) -> str:
    m = int(minutes) if float(minutes).is_integer() else minutes
    return f"{m}_minutes"


@dataclass
class RiskAssessment:
    site_id: str
    prediction_time: str            # timestamp of the latest observation used
    target_time: str                # prediction_time + horizon
    prediction_horizon: str
    congestion_probability: float
    risk_level: str
    predicted_state: str
    confidence: float
    recommended_action: str
    class_probabilities: dict = field(default_factory=dict)
    model: str = ""
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RiskEngine:
    def __init__(self, cfg: dict, model_tau: float):
        lv = cfg["risk"]["levels"]
        medium = float(model_tau) if str(lv["medium"]).lower() == "auto" else float(lv["medium"])
        self.thresholds = [medium, float(lv["high"]), float(lv["critical"])]
        if not (0 < self.thresholds[0] < self.thresholds[1] < self.thresholds[2] <= 1):
            raise ValueError(f"Risk thresholds must satisfy 0 < medium < high < critical <= 1; got {self.thresholds}")
        self.actions = dict(cfg["risk"]["actions"])
        missing = [lv_ for lv_ in RISK_LEVELS if lv_ not in self.actions]
        if missing:
            raise ValueError(f"Missing recommended actions for levels: {missing}")
        lo, hi = (float(x) for x in cfg["risk"].get("probability_bounds", [0.0, 1.0]))
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"risk.probability_bounds must satisfy 0 <= low < high <= 1; got {[lo, hi]}")
        self.p_min, self.p_max = lo, hi
        self.horizon_minutes = float(cfg["prediction"]["horizon_minutes"])
        self.horizon = pd.Timedelta(minutes=self.horizon_minutes)

    def bound(self, p):
        """Clip a calibrated probability to the reportable range (no claims of absolute certainty)."""
        return np.clip(p, self.p_min, self.p_max)

    def level_ids(self, p: np.ndarray) -> np.ndarray:
        """0..3 for LOW..CRITICAL (a probability exactly on a threshold belongs to the higher level)."""
        return np.digitize(np.asarray(p, dtype=float), self.thresholds)

    def level(self, p: float) -> str:
        return RISK_LEVELS[int(self.level_ids(np.array([p]))[0])]

    def assess(self, site_id: str, prediction_time: pd.Timestamp, p_cong: float, class_probs: np.ndarray,
               state_id: int, model_name: str = "", context: dict | None = None) -> RiskAssessment:
        if not 0.0 <= p_cong <= 1.0:
            raise ValueError(f"Probability out of range: {p_cong}")
        pt = pd.Timestamp(prediction_time)
        p_cong = float(self.bound(p_cong))
        level = self.level(p_cong)
        congested = state_id >= 2
        return RiskAssessment(
            site_id=str(site_id), prediction_time=pt.isoformat(), target_time=(pt + self.horizon).isoformat(),
            prediction_horizon=horizon_label(self.horizon_minutes),
            congestion_probability=round(float(p_cong), 4), risk_level=level,
            predicted_state=STATE_NAMES[int(state_id)],
            confidence=round(float(p_cong if congested else 1.0 - p_cong), 4),
            recommended_action=self.actions[level],
            class_probabilities={n: round(float(v), 4) for n, v in zip(STATE_NAMES, class_probs)},
            model=model_name, context=context or {})
