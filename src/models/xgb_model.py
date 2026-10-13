"""XGBoost congestion model: multiclass (4 states) + calibrated congestion probability.

Design
  * objective multi:softprob over NORMAL/WARNING/CONGESTED/CRITICAL
  * congestion probability  P(cong) = P(CONGESTED) + P(CRITICAL)   (one model, two uses)
  * class imbalance: sample weights (n / (4 * class_count)) ** power, power tuned
  * early stopping + hyper-parameter search use the VALIDATION period only
  * optional isotonic calibration of P(cong), fitted on validation
  * alert threshold tau chosen on validation by maximising F-beta (beta=2: a
    missed congestion event costs more than an extra warning)
  * decision: P(cong) >= tau -> CONGESTED/CRITICAL (argmax of the two), else NORMAL/WARNING
The TEST period is never touched here.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, f1_score

from src.config import horizon_steps

logger = logging.getLogger(__name__)
N_CLASSES = 4
SYNTHETIC_NOTE = "Trained on SIMULATED data. Not validated on real telecom measurements."


def sample_weights(y: np.ndarray, power: float) -> np.ndarray:
    """Per-row weights that up-weight rare classes; power=0 -> all ones, power=1 -> balanced."""
    counts = np.bincount(y.astype(int), minlength=N_CLASSES).astype(float)
    cw = (len(y) / (N_CLASSES * np.maximum(counts, 1.0))) ** float(power)
    return cw[y.astype(int)]


def select_threshold(y_bin: np.ndarray, p: np.ndarray, beta: float, grid: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Threshold maximising F-beta of the binary 'congested' decision."""
    rows = []
    for t in grid:
        pred = p >= t
        tp, fp, fn = int((pred & y_bin).sum()), int((pred & ~y_bin).sum()), int((~pred & y_bin).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fb = (1 + beta**2) * prec * rec / (beta**2 * prec + rec) if (prec + rec) else 0.0
        rows.append({"threshold": float(t), "precision": prec, "recall": rec, "fbeta": fb})
    tab = pd.DataFrame(rows)
    return float(tab.loc[tab["fbeta"].idxmax(), "threshold"]), tab


class XGBCongestionModel:
    def __init__(self, cfg: dict):
        self.cfg = copy.deepcopy(cfg)
        self.xc = self.cfg["xgboost"]
        self.model: xgb.XGBClassifier | None = None
        self.feature_columns: list[str] = []
        self.feature_groups: dict[str, str] = {}
        self.params: dict = {}
        self.threshold: float = 0.5
        self.cal_x: np.ndarray | None = None
        self.cal_y: np.ndarray | None = None
        self.meta: dict = {}

    # ------------------------------------------------------------------ fitting
    def _make(self, params: dict) -> xgb.XGBClassifier:
        p = {k: v for k, v in params.items() if k != "class_weight_power"}
        return xgb.XGBClassifier(
            objective="multi:softprob", n_estimators=int(self.xc["n_estimators"]),
            early_stopping_rounds=int(self.xc["early_stopping_rounds"]), eval_metric="mlogloss",
            tree_method="hist", random_state=int(self.cfg["project"]["seed"]),
            n_jobs=self.xc.get("n_jobs"), **p)

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        return df[self.feature_columns].to_numpy(dtype=np.float32)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, feature_columns: list[str],
            params: dict | None = None, stride: int = 1) -> "XGBCongestionModel":
        """Train with early stopping on `val`. `stride` thins the training rows (used by the search)."""
        self.feature_columns = list(feature_columns)
        self.params = {**self.xc["base_params"], **(params or {})}
        tr = train.iloc[::stride]
        y, yv = tr["target_state_id"].to_numpy(int), val["target_state_id"].to_numpy(int)
        power = self.params["class_weight_power"]
        self.model = self._make(self.params)
        t0 = time.time()
        self.model.fit(self._X(tr), y, sample_weight=sample_weights(y, power),
                       eval_set=[(self._X(val), yv)], sample_weight_eval_set=[sample_weights(yv, power)],
                       verbose=False)
        self.meta["fit_seconds"] = round(time.time() - t0, 1)
        self.meta["best_iteration"] = int(self.model.best_iteration)
        self.meta["train_rows"] = int(len(tr))
        return self

    def tune(self, train: pd.DataFrame, val: pd.DataFrame, feature_columns: list[str]) -> tuple[dict, pd.DataFrame]:
        """Random search; objective = validation PR-AUC of P(cong). Trial 0 is the base config."""
        tc = self.xc["tuning"]
        rng = np.random.default_rng(int(tc["seed"]))
        yb = val["target_state_id"].to_numpy() >= 2
        trials, best = [], (-1.0, {})
        for i in range(int(tc["n_iter"])):
            cand = {} if i == 0 else {k: v[int(rng.integers(len(v)))] for k, v in tc["space"].items()}
            m = XGBCongestionModel(self.cfg).fit(train, val, feature_columns, cand, stride=int(tc["train_stride"]))
            p = m.predict_proba(val)
            score = float(average_precision_score(yb, p[:, 2:].sum(1)))
            macro = float(f1_score(val["target_state_id"], p.argmax(1), average="macro"))
            trials.append({"trial": i, **m.params, "val_pr_auc": score, "val_macro_f1_argmax": macro,
                           "best_iteration": m.meta["best_iteration"], "seconds": m.meta["fit_seconds"]})
            logger.info("Trial %d: PR-AUC %.4f (iters %d, %.0fs) %s", i, score, m.meta["best_iteration"],
                        m.meta["fit_seconds"], cand)
            if score > best[0]:
                best = (score, {**m.params})
        return best[1], pd.DataFrame(trials)

    # ------------------------------------------------------------ probabilities
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Raw class probabilities, shape (n, 4)."""
        if self.model is None:
            raise RuntimeError("Model is not trained/loaded.")
        return self.model.predict_proba(self._X(df))

    def raw_congestion_score(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(df)[:, 2:].sum(1)

    def _calibrate_values(self, p: np.ndarray) -> np.ndarray:
        if self.cal_x is None:
            return p
        return np.interp(p, self.cal_x, self.cal_y)

    def congestion_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated P(CONGESTED or CRITICAL at t + horizon)."""
        return self._calibrate_values(self.raw_congestion_score(df))

    def calibrated_probability(self, p: np.ndarray) -> np.ndarray:
        """Calibrated P(CONGESTED or CRITICAL) from a class-probability matrix."""
        return self._calibrate_values(p[:, 2:].sum(1))

    def calibrate(self, val: pd.DataFrame) -> None:
        if str(self.xc.get("calibration", "none")).lower() != "isotonic":
            self.cal_x = self.cal_y = None
            return
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(self.raw_congestion_score(val), (val["target_state_id"].to_numpy() >= 2).astype(float))
        self.cal_x, self.cal_y = iso.X_thresholds_.astype(float), iso.y_thresholds_.astype(float)

    def fit_threshold(self, val: pd.DataFrame) -> pd.DataFrame:
        tc = self.xc["threshold"]
        grid = np.linspace(tc["grid_min"], tc["grid_max"], int(tc["grid_points"]))
        self.threshold, tab = select_threshold(val["target_state_id"].to_numpy() >= 2,
                                               self.congestion_probability(val), float(tc["beta"]), grid)
        return tab

    def decide(self, p: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Turn class probabilities into states using the (calibrated) alert threshold."""
        tau = self.threshold if threshold is None else threshold
        cong = self._calibrate_values(p[:, 2:].sum(1)) >= tau
        return np.where(cong, 2 + p[:, 2:].argmax(1), p[:, :2].argmax(1))

    def predict(self, df: pd.DataFrame, use_threshold: bool = True) -> np.ndarray:
        """States 0..3. With use_threshold the binary decision follows tau (recall-oriented)."""
        p = self.predict_proba(df)
        return self.decide(p) if use_threshold else p.argmax(1)

    # --------------------------------------------------------- interpretation
    def importance(self) -> pd.DataFrame:
        """Gain-based importance per feature (share of total gain) with its feature group."""
        gain = self.model.get_booster().get_score(importance_type="gain")
        rows = [{"feature": self.feature_columns[int(k[1:])], "gain": v} for k, v in gain.items()]
        d = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
        d["gain_share"] = d["gain"] / d["gain"].sum()
        d["group"] = d["feature"].map(self.feature_groups).fillna("other")
        return d

    def shap_top(self, df: pd.DataFrame, k: int = 20, max_rows: int = 3000) -> pd.DataFrame:
        """Mean |TreeSHAP contribution| towards CONGESTED + CRITICAL (native xgboost, no extra library)."""
        s = df.sample(min(len(df), max_rows), random_state=0)
        contrib = self.model.get_booster().predict(xgb.DMatrix(self._X(s)), pred_contribs=True)  # (n, classes, F+1)
        mag = np.abs(contrib[:, 2:, :-1]).sum(1).mean(0)
        return (pd.DataFrame({"feature": self.feature_columns, "mean_abs_shap": mag})
                .sort_values("mean_abs_shap", ascending=False).head(k).reset_index(drop=True))

    # -------------------------------------------------------------- persistence
    def save(self, directory: str | Path, extra_meta: dict | None = None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_model(d / self.xc["model_filename"])
        meta = {
            "model_type": "xgboost_multiclass", "classes": ["NORMAL", "WARNING", "CONGESTED", "CRITICAL"],
            "feature_columns": self.feature_columns, "feature_groups": self.feature_groups,
            "params": self.params, "threshold": self.threshold,
            "threshold_beta": self.xc["threshold"]["beta"],
            "calibration": {"method": "isotonic" if self.cal_x is not None else "none",
                            "x": None if self.cal_x is None else self.cal_x.tolist(),
                            "y": None if self.cal_y is None else self.cal_y.tolist()},
            "horizon_steps": horizon_steps(self.cfg), "sampling_freq": self.cfg["data"]["sampling_freq"],
            "xgboost_version": xgb.__version__, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_note": SYNTHETIC_NOTE, **self.meta, **(extra_meta or {})}
        (d / self.xc["meta_filename"]).write_text(json.dumps(meta, indent=2, default=str))
        return d

    @staticmethod
    def load(directory: str | Path, cfg: dict) -> "XGBCongestionModel":
        d = Path(directory)
        meta_path, model_path = d / cfg["xgboost"]["meta_filename"], d / cfg["xgboost"]["model_filename"]
        if not meta_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"XGBoost artefacts not found in {d}")
        meta = json.loads(meta_path.read_text())
        m = XGBCongestionModel(cfg)
        m.model = xgb.XGBClassifier()
        m.model.load_model(model_path)
        m.feature_columns, m.feature_groups = meta["feature_columns"], meta.get("feature_groups", {})
        m.params, m.threshold, m.meta = meta["params"], float(meta["threshold"]), meta
        cal = meta["calibration"]
        if cal["method"] == "isotonic":
            m.cal_x, m.cal_y = np.array(cal["x"]), np.array(cal["y"])
        return m
