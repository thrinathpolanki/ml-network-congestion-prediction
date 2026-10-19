"""Prediction service core: raw readings -> cleaned history -> features -> model -> risk assessment.

One class serves XGBoost, the LSTM, or the HYBRID (average of both). Training and serving share the SAME
FeatureBuilder (saved with its config, including the labeling thresholds), so features are computed identically.
The model used is read from models/production_model.json (written by the Phase 10 comparison)
unless overridden.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import horizon_steps, resolve_path
from src.features.engineering import FeatureBuilder, InsufficientHistoryError
from src.features.labeling import STATE_NAMES
from src.models.compare import align_probs
from src.models.hybrid import HybridModel
from src.models.lstm_model import LSTMCongestionModel
from src.models.xgb_model import XGBCongestionModel
from src.prediction.history import prepare_history
from src.prediction.risk import RISK_LEVELS, RiskAssessment, RiskEngine

logger = logging.getLogger(__name__)
CONTEXT_COLS = ["bandwidth_utilization_pct", "latency_ms", "packet_loss_pct", "active_users"]
SERVABLE = ("xgboost", "lstm", "hybrid")


class ModelConfigError(RuntimeError):
    """Raised when the saved model and the current configuration disagree."""


class CongestionPredictor:
    def __init__(self, cfg: dict, builder: FeatureBuilder, name: str, model, seq_builder, meta: dict,
                 parts: dict | None = None):
        self.cfg, self.builder, self.name, self.model, self.sb, self.meta = cfg, builder, name, model, seq_builder, meta
        self.parts = parts or {}                       # {"xgboost": ..., "lstm": ...} for the hybrid
        self.step = pd.Timedelta(cfg["data"]["sampling_freq"])
        self.risk = RiskEngine(cfg, model.threshold)
        # the hybrid needs XGBoost's longer history; the LSTM alone needs only its window
        self.required_steps = seq_builder.seq_len if name == "lstm" else builder.required_history_steps

    # ------------------------------------------------------------------ loading
    @staticmethod
    def load(cfg: dict, models_dir: str | Path | None = None, model: str | None = None) -> "CongestionPredictor":
        d = Path(models_dir) if models_dir else resolve_path(cfg, "models_dir")
        choice = (model or cfg["serving"]["model"]).lower()
        if choice == "auto":
            pm = d / "production_model.json"
            choice = json.loads(pm.read_text())["model"] if pm.exists() else cfg["serving"]["default_model"]
            logger.info("Serving model from %s: %s", "production_model.json" if pm.exists() else "config default", choice)
        if choice not in SERVABLE:
            raise ValueError(f"Unknown model '{choice}' (servable: {', '.join(SERVABLE)})")
        fb_path = d / cfg["features"]["builder_filename"]
        if not fb_path.exists():
            raise FileNotFoundError(f"Feature builder not found: {fb_path}. Run scripts/build_features.py")
        builder = FeatureBuilder.load(fb_path)
        sb, parts = None, None
        if choice == "xgboost":
            m = XGBCongestionModel.load(d, cfg)
        elif choice == "lstm":
            m, sb = LSTMCongestionModel.load(d, cfg)
        else:
            xm = XGBCongestionModel.load(d, cfg)
            lm, sb = LSTMCongestionModel.load(d, cfg)
            m, parts = HybridModel.load(d), {"xgboost": xm, "lstm": lm}
        expected = horizon_steps(cfg)
        for label, mm in ({"hybrid": m, **parts}.items() if parts else {choice: m}.items()):
            got = int(mm.meta.get("horizon_steps", expected))
            if got != expected:
                raise ModelConfigError(f"The {label} model was trained for a horizon of {got} steps but the config now "
                                       f"implies {expected}. Retrain the models for the new horizon.")
        pred = CongestionPredictor(cfg, builder, choice, m, sb, m.meta, parts)
        logger.info("Predictor ready: model=%s, horizon=%d steps, tau=%.3f, history=%d steps",
                    choice, expected, m.threshold, pred.required_steps)
        return pred

    # ---------------------------------------------------------------- inference
    def _p_xgb(self, xm, tail: pd.DataFrame) -> np.ndarray:
        return xm.predict_proba(self.builder.transform_latest(tail))[0]

    def _p_lstm(self, lm, tail: pd.DataFrame) -> np.ndarray:
        f = self.builder.transform(tail.tail(self.sb.seq_len), drop_incomplete=False)
        dyn, stat = self.sb.window_from_features(f)
        return lm.proba_from_arrays(dyn, stat)[0]

    def _proba_latest(self, tail: pd.DataFrame) -> np.ndarray:
        if self.name == "xgboost":
            return self._p_xgb(self.model, tail)
        if self.name == "lstm":
            return self._p_lstm(self.model, tail)
        # hybrid: build the features ONCE and give both models what they need (the LSTM's inputs are row-wise,
        # so they are identical whether computed from 36 or 73 rows; tests/test_hybrid.py proves equivalence)
        feats = self.builder.transform(tail, drop_incomplete=False)
        px = self.parts["xgboost"].predict_proba(self.builder.latest_row(feats))[0]
        dyn, stat = self.sb.window_from_features(feats)
        return HybridModel.blend(px, self.parts["lstm"].proba_from_arrays(dyn, stat)[0])

    def _context(self, tail: pd.DataFrame) -> dict:
        last = tail.iloc[-1]
        ctx = {c: (None if pd.isna(last[c]) else round(float(last[c]), 3)) for c in CONTEXT_COLS}
        ago = tail["bandwidth_utilization_pct"].iloc[-7] if len(tail) >= 7 else np.nan
        ctx["utilization_change_30min_pct_points"] = (None if pd.isna(ago) or pd.isna(last["bandwidth_utilization_pct"])
                                                      else round(float(last["bandwidth_utilization_pct"] - ago), 2))
        ctx["region"] = str(last["region"])
        for c in ("latitude", "longitude"):                      # optional cell position, for maps; absent if the source has none
            if c in tail.columns and pd.notna(last[c]):
                ctx[c] = round(float(last[c]), 6)
        if "location" in tail.columns and pd.notna(last["location"]):
            ctx["location"] = str(last["location"])
        return ctx

    def predict_site(self, history: pd.DataFrame) -> RiskAssessment:
        """Assess ONE site from its recent raw readings (>= `required_steps` consecutive intervals)."""
        prep = prepare_history(history, self.cfg)
        if prep["site_id"].nunique() != 1:
            raise ValueError("predict_site expects readings for exactly one site; use predict_many.")
        if len(prep) < self.required_steps:
            raise InsufficientHistoryError(
                f"Need >= {self.required_steps} consecutive readings ({self.required_steps * self.step}); "
                f"got {len(prep)} after cleaning.")
        tail = prep.tail(self.required_steps)
        p = self._proba_latest(tail)
        pc = float(self.model.calibrated_probability(p[None, :])[0])
        state = int(self.model.decide(p[None, :])[0])
        return self.risk.assess(tail["site_id"].iloc[-1], tail["timestamp"].iloc[-1], pc, p, state,
                                self.name, self._context(tail))

    def predict_many(self, history: pd.DataFrame) -> tuple[list[RiskAssessment], dict[str, str]]:
        """Assess every site in `history`. One bad site never blocks the others: errors are returned per site."""
        out, errors = [], {}
        for site, g in history.groupby("site_id", sort=True):
            try:
                out.append(self.predict_site(g))
            except (InsufficientHistoryError, ValueError) as e:
                errors[str(site)] = str(e)
                logger.warning("Site %s skipped: %s", site, e)
        return out, errors

    def _frame_proba(self, feats: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """(keys, class probabilities) for every row of a feature frame that the model can score."""
        cols = self.builder.feature_columns

        def xgb_part(xm):
            rows = feats[feats[cols].notna().all(axis=1)].reset_index(drop=True)
            return rows[["site_id", "timestamp", "state_now"]], xm.predict_proba(rows)

        def lstm_part(lm):
            ss = self.sb.build(feats)
            return ss.meta[["site_id", "timestamp", "state_now"]], lm.predict_proba(ss)

        if self.name == "xgboost":
            return xgb_part(self.model)
        if self.name == "lstm":
            return lstm_part(self.model)
        kx, px = xgb_part(self.parts["xgboost"])
        kl, pl = lstm_part(self.parts["lstm"])
        keys, px, pl = align_probs(kx, px, kl, pl)
        return keys, HybridModel.blend(px, pl)

    def predict_frame(self, history: pd.DataFrame) -> pd.DataFrame:
        """Backtest view: a prediction for every timestamp with enough history (used by the dashboard)."""
        prep = prepare_history(history, self.cfg)
        feats = self.builder.transform(prep, drop_incomplete=False)
        keys, p = self._frame_proba(feats)
        pc = self.risk.bound(self.model.calibrated_probability(p))
        state = self.model.decide(p)
        out = keys.reset_index(drop=True).rename(columns={"timestamp": "prediction_time"})
        out["target_time"] = out["prediction_time"] + self.risk.horizon
        out["congestion_probability"] = pc
        out["risk_level"] = np.array(RISK_LEVELS)[self.risk.level_ids(pc)]
        out["predicted_state"] = np.array(STATE_NAMES)[state]
        return out

    def info(self) -> dict:
        return {"model": self.name, "components": list(self.parts) or None, "horizon_minutes": self.risk.horizon_minutes,
                "alert_threshold_tau": round(float(self.model.threshold), 4),
                "risk_thresholds": dict(zip(RISK_LEVELS[1:], [round(t, 4) for t in self.risk.thresholds])),
                "history_required_steps": self.required_steps,
                "history_required_minutes": self.required_steps * self.step.total_seconds() / 60,
                "sampling_interval": self.cfg["data"]["sampling_freq"],
                "train_period": (self.parts.get("lstm").meta if self.parts else self.meta).get("train_period"),
                "val_period": self.meta.get("val_period"),
                "data_note": self.meta.get("data_note", "")}
