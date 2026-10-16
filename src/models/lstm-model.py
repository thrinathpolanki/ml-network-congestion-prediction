"""LSTM congestion model: past N intervals of KPIs -> congestion state at t + horizon.

Inputs per sample
  dynamic sequence  (N steps x 25): 15 current KPIs, current score, time-of-day / day-of-week
                    encodings, weekend + peak flags, 3 site-relative KPIs
  static vector     (13): site capacity + baselines, region one-hot, calendar flags of the
                    FORECAST time t+H (known in advance)
Architecture   LSTM(25 -> hidden) -> last hidden state ++ static -> MLP -> 4 logits
Same decision policy as XGBoost: P(cong) = P(CONGESTED)+P(CRITICAL), isotonic calibration and
an F2-optimal alert threshold, all fitted on VALIDATION only.

Leakage control
  * scaler fitted on training-period rows only
  * windows are built from a regular per-site grid, never crossing sites; each window ends at t
  * windows containing missing values are dropped (not imputed)
  * the TEST period is never touched here
"""
from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from src.config import horizon_steps
from src.features.engineering import InsufficientHistoryError
from src.models.xgb_model import SYNTHETIC_NOTE, select_threshold
from src.preprocessing.scaling import FeatureTransformer

logger = logging.getLogger(__name__)
KEYS = ["site_id", "timestamp"]
N_CLASSES = 4
DYN_EXTRA = ["score_now", "tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "is_peak_hour"]


def column_sets(builder) -> tuple[list[str], list[str]]:
    """(dynamic, static) feature columns taken from a fitted FeatureBuilder."""
    g = builder.feature_groups
    dyn = list(g["current_kpis"]) + DYN_EXTRA + list(g["site_relative"])
    static = list(g["site_static"]) + list(g["region_onehot"]) + list(g["calendar_forecast_time"])
    return dyn, static


# --------------------------------------------------------------------------- #
# Sequence construction
# --------------------------------------------------------------------------- #
@dataclass
class SequenceSet:
    """Whole-timeline arrays plus the end positions of the windows that form this split."""
    dyn: np.ndarray           # (N_all, F) standardised, per-site contiguous, regular grid
    stat: np.ndarray          # (N_all, S)
    end_idx: np.ndarray       # (n,) positions of each window's LAST step
    meta: pd.DataFrame        # n rows: site_id, timestamp, state_now, [target_state_id]
    seq_len: int

    def __len__(self) -> int:
        return len(self.end_idx)

    @property
    def y(self) -> np.ndarray:
        return self.meta["target_state_id"].to_numpy(int)

    def arrays(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = self.end_idx[rows]
        idx = e[:, None] + np.arange(-self.seq_len + 1, 1)[None, :]
        return self.dyn[idx], self.stat[e]


class SequenceBuilder:
    """Fits the scaler on the training period and turns feature frames into windows."""

    def __init__(self, dyn_cols: list[str], static_cols: list[str], seq_len: int):
        self.dyn_cols, self.static_cols, self.seq_len = list(dyn_cols), list(static_cols), int(seq_len)
        self.scaler = FeatureTransformer(self.dyn_cols + self.static_cols)

    def fit(self, train_feats: pd.DataFrame) -> "SequenceBuilder":
        self.scaler.fit(train_feats)
        return self

    def _scaled(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        z = self.scaler.transform(feats)
        return (z[self.dyn_cols].to_numpy(np.float32), z[self.static_cols].to_numpy(np.float32))

    def build(self, feats_all: pd.DataFrame, rows: pd.DataFrame | None = None) -> SequenceSet:
        """feats_all: output of FeatureBuilder.transform(..., drop_incomplete=False) (sorted, regular grid).
        rows: split table whose (site, timestamp) rows become samples (with target_state_id)."""
        f = feats_all.sort_values(KEYS).reset_index(drop=True)
        dyn, stat = self._scaled(f)
        L = self.seq_len
        pos = f.groupby("site_id").cumcount().to_numpy()
        bad = np.isnan(dyn).any(1).astype(np.int64)
        c = np.concatenate([[0], np.cumsum(bad)])                       # c[i+1] = bad rows in [0..i]
        i = np.arange(len(f))
        window_bad = c[i + 1] - c[np.maximum(i - L + 1, 0)]
        ok = (pos >= L - 1) & (window_bad == 0)
        index = pd.MultiIndex.from_frame(f[KEYS])
        if rows is None:
            end = np.flatnonzero(ok)
            meta = f.loc[end, KEYS + ["state_now"]].reset_index(drop=True)
        else:
            want = rows.reset_index(drop=True)
            end_all = index.get_indexer(pd.MultiIndex.from_frame(want[KEYS]))
            keep = (end_all >= 0)
            keep[keep] &= ok[end_all[keep]]
            end, meta = end_all[keep], want.loc[keep, [c for c in KEYS + ["state_now", "target_state_id"] if c in want.columns]].reset_index(drop=True)
            if (~keep).any():
                logger.info("Dropped %d of %d rows (insufficient / incomplete window)", int((~keep).sum()), len(want))
        return SequenceSet(dyn, stat, end.astype(np.int64), meta, L)

    def window_from_features(self, feats_site: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Single window from the LAST seq_len rows of one site (used at inference time)."""
        if feats_site["site_id"].nunique() != 1:
            raise ValueError("Expected the history of exactly one site.")
        f = feats_site.sort_values("timestamp")
        if len(f) < self.seq_len:
            raise InsufficientHistoryError(f"LSTM needs >= {self.seq_len} consecutive readings; got {len(f)}.")
        dyn, stat = self._scaled(f.tail(self.seq_len))
        if np.isnan(dyn).any():
            raise InsufficientHistoryError("History window contains missing KPIs.")
        return dyn[None, :, :], stat[-1:][:]


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class CongestionNet(nn.Module):
    def __init__(self, n_dyn: int, n_static: int, hidden: int = 64, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_dyn, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, N_CLASSES))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(torch.cat([h[-1], s], dim=1))


def _class_weights(y: np.ndarray, power: float) -> torch.Tensor:
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    return torch.tensor((len(y) / (N_CLASSES * np.maximum(counts, 1))) ** power, dtype=torch.float32)


class LSTMCongestionModel:
    def __init__(self, cfg: dict, dyn_cols: list[str], static_cols: list[str]):
        self.cfg, self.lc = copy.deepcopy(cfg), cfg["lstm"]
        self.dyn_cols, self.static_cols = list(dyn_cols), list(static_cols)
        if self.lc.get("num_threads"):
            torch.set_num_threads(int(self.lc["num_threads"]))
        torch.manual_seed(int(cfg["project"]["seed"]))
        self.net = CongestionNet(len(dyn_cols), len(static_cols), int(self.lc["hidden_size"]),
                                 int(self.lc["num_layers"]), float(self.lc["dropout"]))
        self.threshold, self.cal_x, self.cal_y = 0.5, None, None
        self.history: pd.DataFrame | None = None
        self.meta: dict = {}

    @property
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    # ---- training
    def _logits(self, s: SequenceSet, rows: np.ndarray) -> torch.Tensor:
        x, st = s.arrays(rows)
        return self.net(torch.from_numpy(x), torch.from_numpy(st))

    def _eval(self, s: SequenceSet, cw: torch.Tensor | None = None, bs: int = 2048):
        self.net.eval()
        probs, loss_sum, w_sum = [], 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(s), bs):
                rows = np.arange(i, min(i + bs, len(s)))
                lg = self._logits(s, rows)
                probs.append(F.softmax(lg, 1).numpy())
                if cw is not None:
                    y = torch.from_numpy(s.y[rows])
                    ce = F.cross_entropy(lg, y, reduction="none")
                    loss_sum += float((ce * cw[y]).sum()); w_sum += float(cw[y].sum())
        return np.concatenate(probs), (loss_sum / w_sum if cw is not None else None)

    def fit(self, train: SequenceSet, val: SequenceSet, checkpoint_path: str | Path | None = None) -> pd.DataFrame:
        lc = self.lc
        torch.manual_seed(int(self.cfg["project"]["seed"]))   # dropout draws from the global RNG
        cw = _class_weights(train.y, float(lc["class_weight_power"]))
        loss_fn = nn.CrossEntropyLoss(weight=cw)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(lc["learning_rate"]),
                               weight_decay=float(lc["weight_decay"]))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
        rng = np.random.default_rng(int(self.cfg["project"]["seed"]))
        yb_val = val.y >= 2
        best, best_state, bad, hist = np.inf, None, 0, []
        bs = int(lc["batch_size"])
        for epoch in range(1, int(lc["max_epochs"]) + 1):
            t0 = time.time()
            self.net.train()
            perm = rng.permutation(len(train))
            if lc.get("windows_per_epoch"):
                perm = perm[: int(lc["windows_per_epoch"])]
            tl, nb = 0.0, 0
            for i in range(0, len(perm), bs):
                rows = np.sort(perm[i:i + bs])
                lg = self._logits(train, rows)
                loss = loss_fn(lg, torch.from_numpy(train.y[rows]))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), float(lc["grad_clip"]))
                opt.step()
                tl += loss.item(); nb += 1
            p, vl = self._eval(val, cw)
            pr = float(average_precision_score(yb_val, p[:, 2:].sum(1)))
            sched.step(vl)
            hist.append({"epoch": epoch, "train_loss": tl / nb, "val_loss": vl, "val_pr_auc": pr,
                         "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)})
            logger.info("Epoch %2d | train %.4f | val %.4f | val PR-AUC %.4f | %.0fs", epoch, tl / nb, vl, pr, time.time() - t0)
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = copy.deepcopy(self.net.state_dict())
                if checkpoint_path:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state_dict": best_state, "epoch": epoch, "val_loss": vl}, checkpoint_path)
            else:
                bad += 1
                if bad >= int(lc["patience"]):
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        self.net.load_state_dict(best_state)
        self.history = pd.DataFrame(hist)
        self.meta.update(best_epoch=int(self.history.loc[self.history.val_loss.idxmin(), "epoch"]),
                         epochs_run=len(hist), train_seconds=round(float(self.history.seconds.sum()), 1),
                         n_parameters=self.n_parameters, train_windows=len(train))
        return self.history

    # ---- probabilities and decisions (same policy as XGBoost)
    def predict_proba(self, s: SequenceSet) -> np.ndarray:
        return self._eval(s)[0]

    def raw_congestion_score(self, s: SequenceSet) -> np.ndarray:
        return self.predict_proba(s)[:, 2:].sum(1)

    def _cal(self, p: np.ndarray) -> np.ndarray:
        return p if self.cal_x is None else np.interp(p, self.cal_x, self.cal_y)

    def calibrated_probability(self, p: np.ndarray) -> np.ndarray:
        """Calibrated P(CONGESTED or CRITICAL) from a class-probability matrix."""
        return self._cal(p[:, 2:].sum(1))

    def calibrate(self, val: SequenceSet) -> None:
        if str(self.lc.get("calibration", "none")).lower() != "isotonic":
            self.cal_x = self.cal_y = None
            return
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(self.raw_congestion_score(val), (val.y >= 2).astype(float))
        self.cal_x, self.cal_y = iso.X_thresholds_.astype(float), iso.y_thresholds_.astype(float)

    def fit_threshold(self, val: SequenceSet) -> pd.DataFrame:
        t = self.lc["threshold"]
        grid = np.linspace(t["grid_min"], t["grid_max"], int(t["grid_points"]))
        p = self._cal(self.raw_congestion_score(val))
        self.threshold, tab = select_threshold(val.y >= 2, p, float(t["beta"]), grid)
        return tab

    def congestion_probability(self, s: SequenceSet) -> np.ndarray:
        return self._cal(self.raw_congestion_score(s))

    def decide(self, p: np.ndarray, threshold: float | None = None) -> np.ndarray:
        tau = self.threshold if threshold is None else threshold
        cong = self._cal(p[:, 2:].sum(1)) >= tau
        return np.where(cong, 2 + p[:, 2:].argmax(1), p[:, :2].argmax(1))

    def predict(self, s: SequenceSet, use_threshold: bool = True) -> np.ndarray:
        p = self.predict_proba(s)
        return self.decide(p) if use_threshold else p.argmax(1)

    def proba_from_arrays(self, dyn: np.ndarray, stat: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            x, st = torch.from_numpy(np.array(dyn, dtype=np.float32)), torch.from_numpy(np.array(stat, dtype=np.float32))
            return F.softmax(self.net(x, st), 1).numpy()

    # ---- persistence
    def save(self, directory: str | Path, seq_builder: SequenceBuilder, extra_meta: dict | None = None) -> Path:
        d, lc = Path(directory), self.lc
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(),
                    "hparams": {"n_dyn": len(self.dyn_cols), "n_static": len(self.static_cols),
                                "hidden": int(lc["hidden_size"]), "layers": int(lc["num_layers"]),
                                "dropout": float(lc["dropout"])}}, d / lc["model_filename"])
        seq_builder.scaler.save(d / lc["scaler_filename"])
        meta = {"model_type": "lstm", "classes": ["NORMAL", "WARNING", "CONGESTED", "CRITICAL"],
                "dynamic_columns": self.dyn_cols, "static_columns": self.static_cols, "seq_len": seq_builder.seq_len,
                "threshold": self.threshold, "threshold_beta": lc["threshold"]["beta"],
                "calibration": {"method": "isotonic" if self.cal_x is not None else "none",
                                "x": None if self.cal_x is None else self.cal_x.tolist(),
                                "y": None if self.cal_y is None else self.cal_y.tolist()},
                "horizon_steps": horizon_steps(self.cfg), "sampling_freq": self.cfg["data"]["sampling_freq"],
                "torch_version": torch.__version__, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "data_note": SYNTHETIC_NOTE, **self.meta, **(extra_meta or {}),
                "history": None if self.history is None else self.history.to_dict(orient="records")}
        (d / lc["meta_filename"]).write_text(json.dumps(meta, indent=2, default=str))
        return d

    @staticmethod
    def load(directory: str | Path, cfg: dict) -> tuple["LSTMCongestionModel", SequenceBuilder]:
        d, lc = Path(directory), cfg["lstm"]
        paths = [d / lc[k] for k in ("model_filename", "meta_filename", "scaler_filename")]
        if not all(p.exists() for p in paths):
            raise FileNotFoundError(f"LSTM artefacts not found in {d}")
        meta = json.loads(paths[1].read_text())
        ck = torch.load(paths[0], map_location="cpu", weights_only=True)
        h = ck["hparams"]                                   # architecture is defined by the checkpoint, not by today's config
        c = copy.deepcopy(cfg)
        c["lstm"].update(hidden_size=h["hidden"], num_layers=h["layers"], dropout=h["dropout"])
        m = LSTMCongestionModel(c, meta["dynamic_columns"], meta["static_columns"])
        m.net.load_state_dict(ck["state_dict"])
        m.threshold, m.meta = float(meta["threshold"]), meta
        if meta["calibration"]["method"] == "isotonic":
            m.cal_x, m.cal_y = np.array(meta["calibration"]["x"]), np.array(meta["calibration"]["y"])
        sb = SequenceBuilder(meta["dynamic_columns"], meta["static_columns"], meta["seq_len"])
        sb.scaler = FeatureTransformer.load(paths[2])
        return m, sb
