"""Fit the HYBRID model's calibration + alert threshold on the VALIDATION period and save models/hybrid_meta.json.

    python scripts/fit_hybrid.py
Needs the trained XGBoost and LSTM in models/ (scripts/train_xgboost.py, scripts/train_lstm.py).
Only validation data is used; the test set is not touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, horizon_steps, resolve_path  # noqa: E402
from src.features.dataset import build_dataset  # noqa: E402
from src.models.compare import OperatingPoint, align_probs  # noqa: E402
from src.models.evaluate import build_frame, evaluate_frame  # noqa: E402
from src.models.hybrid import HybridModel  # noqa: E402
from src.models.lstm_model import LSTMCongestionModel  # noqa: E402
from src.models.xgb_model import XGBCongestionModel  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    cfg = get_config()
    setup_logging(cfg)
    models_dir = resolve_path(cfg, "models_dir")
    clean = resolve_path(cfg, "processed_dir") / cfg["preprocessing"]["clean_filename"]
    if not clean.exists():
        sys.exit("Run scripts/prepare_data.py first.")
    xm = XGBCongestionModel.load(models_dir, cfg)
    lm, sb = LSTMCongestionModel.load(models_dir, cfg)
    df = pd.read_csv(clean, parse_dates=["timestamp"])
    b = build_dataset(df, cfg)
    feats = b.builder.transform(df, drop_incomplete=False)
    vs = sb.build(feats, b.val)
    meta_x = b.val[["site_id", "timestamp", "target_state_id", "state_now"]].reset_index(drop=True)
    meta, px, pl = align_probs(meta_x, xm.predict_proba(b.val), vs.meta, lm.predict_proba(vs))
    p = HybridModel.blend(px, pl)
    y = meta["target_state_id"].to_numpy()
    op = OperatingPoint.fit(p, y, cfg["lstm"]["threshold"])
    model = HybridModel(op, {"horizon_steps": horizon_steps(cfg), "sampling_freq": cfg["data"]["sampling_freq"],
                             "val_rows": int(len(meta)), "val_period": [str(meta.timestamp.min()), str(meta.timestamp.max())],
                             "xgboost_tau": xm.threshold, "lstm_tau": lm.threshold,
                             "val_metrics_note": "calibration and threshold are fitted on validation, so validation metrics are optimistic"})
    path = model.save(models_dir)
    r = evaluate_frame(build_frame(meta, op.decide(p), p[:, 2:].sum(1), p), cfg)
    print(f"\nHybrid operating point fitted on {len(meta):,} validation rows (SIMULATED DATA)")
    print(f"  tau = {op.tau:.3f}  | PR-AUC {average_precision_score(y >= 2, p[:, 2:].sum(1)):.4f}")
    print(f"  recall {r['cong_recall']:.3f} | precision {r['cong_precision']:.3f} | new-onset recall {r['recall_new_onset']:.3f} | "
          f"event recall {r['event_recall']:.3f} | false alerts/site-day {r['false_alert_episodes_per_site_day']:.2f}")
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
