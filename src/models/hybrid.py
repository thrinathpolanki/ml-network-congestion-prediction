"""Hybrid model: the average of the XGBoost and LSTM class-probability vectors.

It has its OWN calibration and alert threshold, fitted on the VALIDATION period exactly like every other
model (src/models/compare.OperatingPoint), and stored in models/hybrid_meta.json. The two component
models are loaded from their own artefacts; nothing here retrains them.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.models.compare import OperatingPoint
from src.models.xgb_model import SYNTHETIC_NOTE

META_FILENAME = "hybrid_meta.json"


class HybridModel:
    """Duck-typed like XGBCongestionModel / LSTMCongestionModel for the parts the predictor uses."""

    def __init__(self, op: OperatingPoint, meta: dict):
        self.op, self.meta = op, meta
        self.threshold = float(op.tau)

    def calibrated_probability(self, p: np.ndarray) -> np.ndarray:
        return self.op.prob(p)

    def decide(self, p: np.ndarray, threshold: float | None = None) -> np.ndarray:
        return self.op.decide(p, threshold)

    @staticmethod
    def blend(p_xgb: np.ndarray, p_lstm: np.ndarray) -> np.ndarray:
        """Equal-weight average of the two 4-class probability vectors (rows still sum to 1)."""
        return 0.5 * (np.asarray(p_xgb, float) + np.asarray(p_lstm, float))

    # ---- persistence
    def save(self, directory: str | Path, extra: dict | None = None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"model_type": "hybrid_average", "components": ["xgboost", "lstm"], "weights": [0.5, 0.5],
                "threshold": self.threshold,
                "calibration": {"method": "isotonic" if self.op.cal_x is not None else "none",
                                "x": None if self.op.cal_x is None else self.op.cal_x.tolist(),
                                "y": None if self.op.cal_y is None else self.op.cal_y.tolist()},
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "data_note": SYNTHETIC_NOTE, **self.meta, **(extra or {})}
        (d / META_FILENAME).write_text(json.dumps(meta, indent=2, default=str))
        return d / META_FILENAME

    @staticmethod
    def load(directory: str | Path) -> "HybridModel":
        p = Path(directory) / META_FILENAME
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Fit the hybrid operating point first: python scripts/fit_hybrid.py")
        meta = json.loads(p.read_text())
        cal = meta["calibration"]
        op = OperatingPoint(np.array(cal["x"]) if cal["x"] is not None else None,
                            np.array(cal["y"]) if cal["y"] is not None else None, float(meta["threshold"]))
        return HybridModel(op, meta)
