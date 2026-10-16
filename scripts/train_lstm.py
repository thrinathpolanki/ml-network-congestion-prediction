"""CLI: train the LSTM on the training period; evaluate on VALIDATION next to XGBoost (test untouched).

    python scripts/train_lstm.py
    python scripts/train_lstm.py --quick          # 3 short epochs, smoke test
    python scripts/train_lstm.py --figures-only   # reload saved model, redo evaluation/figures
Needs: python scripts/prepare_data.py   (and, for the XGBoost comparison, scripts/train_xgboost.py)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, resolve_path  # noqa: E402
from src.features.dataset import build_dataset  # noqa: E402
from src.models.baselines import MajorityClass, Persistence, ThresholdDetector  # noqa: E402
from src.models.evaluate import build_frame, compare_frames, evaluate_frame  # noqa: E402
from src.models.lstm_model import (LSTMCongestionModel, SequenceBuilder, SequenceSet,  # noqa: E402
                                   column_sets)
from src.models.xgb_model import XGBCongestionModel  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

SHOW = ["model", "accuracy", "macro_f1", "cong_precision", "cong_recall", "pr_auc",
        "recall_new_onset", "event_recall", "false_alerts_per_site_day"]
TAUS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def sweep(decide, p, meta, raw, cfg) -> pd.DataFrame:
    rows = []
    for t in TAUS:
        r = evaluate_frame(build_frame(meta.assign(), decide(p, t), raw), cfg)
        rows.append({"tau": t, "precision": r["cong_precision"], "recall": r["cong_recall"],
                     "new_onset_recall": r["recall_new_onset"], "event_recall": r["event_recall"],
                     "false_alerts_per_site_day": r["false_alert_episodes_per_site_day"]})
    return pd.DataFrame(rows)


def figures(hist, curves, fig_dir):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ax[0].plot(hist.epoch, hist.train_loss, "o-", label="train (weighted CE)")
    ax[0].plot(hist.epoch, hist.val_loss, "o-", label="validation")
    ax[0].axvline(hist.loc[hist.val_loss.idxmin(), "epoch"], ls="--", color="grey")
    ax[0].set_xlabel("epoch"); ax[0].set_title("Loss (dashed = checkpoint kept)"); ax[0].legend()
    ax[1].plot(hist.epoch, hist.val_pr_auc, "o-", color="#c0392b"); ax[1].set_xlabel("epoch"); ax[1].set_title("Validation PR-AUC (congested)")
    for name, (yb, score, sw, col) in curves.items():
        pr, rc, _ = precision_recall_curve(yb, score)
        ax[2].plot(rc, pr, color=col, label=name)
    ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision"); ax[2].set_title("Precision-recall on validation"); ax[2].legend()
    fig.tight_layout(); fig.savefig(fig_dir / "15_lstm_training_and_pr.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, (yb, score, sw, col) in curves.items():
        ax.plot(sw["false_alerts_per_site_day"], sw["event_recall"], "o-", color=col, label=name)
        for t, x, y in zip(sw.tau, sw.false_alerts_per_site_day, sw.event_recall):
            ax.annotate(f"{t:g}", (x, y), fontsize=6, xytext=(2, 3), textcoords="offset points")
    ax.set_xlabel("false alert episodes per site-day"); ax.set_ylabel("event recall (episodes flagged early)")
    ax.set_title("Operator trade-off on VALIDATION (labels = tau; simulated data)"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "16_lstm_vs_xgb_operator_val.png", dpi=110); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    cfg = get_config()
    if a.quick:
        cfg["lstm"].update(max_epochs=3, windows_per_epoch=20000)
    setup_logging(cfg)
    set_global_seed(cfg["project"]["seed"])
    models_dir, reports = resolve_path(cfg, "models_dir"), resolve_path(cfg, "reports_dir")
    fig_dir = reports / "figures"; fig_dir.mkdir(exist_ok=True)
    clean_path = resolve_path(cfg, "processed_dir") / cfg["preprocessing"]["clean_filename"]
    if not clean_path.exists():
        sys.exit("Clean data missing. Run: python scripts/prepare_data.py")

    t0 = time.time()
    clean = pd.read_csv(clean_path, parse_dates=["timestamp"])
    bundle = build_dataset(clean, cfg)
    feats_all = bundle.builder.transform(clean, drop_incomplete=False)
    t1 = pd.Timestamp(bundle.report["baselines_fitted_on_rows_before"])
    print(f"Features ready in {time.time() - t0:.0f}s")

    if a.figures_only:
        model, sb = LSTMCongestionModel.load(models_dir, cfg)
        history = pd.DataFrame(model.meta["history"])
    else:
        dyn, static = column_sets(bundle.builder)
        sb = SequenceBuilder(dyn, static, cfg["lstm"]["seq_len"]).fit(feats_all[feats_all.timestamp < t1])
        model = LSTMCongestionModel(cfg, dyn, static)
    train = sb.build(feats_all, bundle.train)
    val = sb.build(feats_all, bundle.val)
    print(f"Windows: train {len(train):,} | val {len(val):,} | seq_len {sb.seq_len} | "
          f"{len(sb.dyn_cols)} dynamic + {len(sb.static_cols)} static inputs | {model.n_parameters:,} parameters")

    if not a.figures_only:
        print("\nTraining (early stopping on weighted validation loss)...")
        history = model.fit(train, val, checkpoint_path=models_dir / cfg["lstm"]["checkpoint_filename"])
        print(history.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        model.calibrate(val)
    tab = model.fit_threshold(val)

    p = model.predict_proba(val)
    raw = p[:, 2:].sum(1)
    frames = {}
    for b in (MajorityClass(), ThresholdDetector(cfg["baselines"]["threshold_feature"]), Persistence()):
        b.fit(bundle.train)
        frames[b.name] = build_frame(bundle.val, b.predict(bundle.val), b.congestion_score(bundle.val))
    curves_src = {}
    try:
        xgbm = XGBCongestionModel.load(models_dir, cfg)
        px = xgbm.predict_proba(bundle.val)
        frames["XGBoost (argmax)"] = build_frame(bundle.val, px.argmax(1), px[:, 2:].sum(1), px)
        frames["XGBoost (tuned tau)"] = build_frame(bundle.val, xgbm.decide(px), px[:, 2:].sum(1), px)
        curves_src["XGBoost"] = (bundle.val, xgbm.decide, px, "#2c3e50")
    except FileNotFoundError:
        print("(XGBoost model not found - run scripts/train_xgboost.py for the side-by-side comparison)")
    frames["LSTM (argmax)"] = build_frame(val.meta, p.argmax(1), raw, p)
    frames["LSTM (tuned tau)"] = build_frame(val.meta, model.decide(p), raw, p)
    curves_src["LSTM"] = (val.meta, model.decide, p, "#c0392b")
    table, results, _ = compare_frames(frames, cfg)
    print(f"\n=== VALIDATION, common rows ({results['Persistence (state now)']['n_rows']:,}; SIMULATED DATA; val used for early stopping/calibration/tau => optimistic) ===")
    print(table[SHOW].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(table[["model", "macro_f1_ci95", "cong_recall_ci95"]].to_string(index=False))

    curves, ops = {}, {}
    for name, (meta, decide, pp, col) in curves_src.items():
        sw = sweep(decide, pp, meta, pp[:, 2:].sum(1), cfg)
        ops[name] = sw
        curves[name] = (meta["target_state_id"].to_numpy() >= 2, pp[:, 2:].sum(1), sw, col)
    print("\nOperating points (calibrated P(cong) >= tau), LSTM:")
    print(ops["LSTM"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nChosen tau = {model.threshold:.2f}")

    sub = SequenceSet(val.dyn, val.stat, val.end_idx[:5000], val.meta.head(5000), val.seq_len)
    t = time.perf_counter(); model.predict(sub); ms = (time.perf_counter() - t) / len(sub) * 1e6
    print(f"Inference: {ms:.0f} ms per 1,000 windows (includes window gathering)")
    figures(history, curves, fig_dir)

    if not a.figures_only:
        model.save(models_dir, sb, extra_meta={
            "train_period": [str(bundle.train.timestamp.min()), str(bundle.train.timestamp.max())],
            "val_period": [str(bundle.val.timestamp.min()), str(bundle.val.timestamp.max())],
            "val_metrics_note": "validation metrics are optimistic (early stopping, calibration, tau all use validation)",
            "inference_ms_per_1000_rows": round(ms, 1), "quick_run": bool(a.quick)})
    (reports / "lstm_report.json").write_text(json.dumps({
        "val_results": results, "history": history.to_dict(orient="records"),
        "operating_points": {k: v.to_dict(orient="records") for k, v in ops.items()},
        "threshold": model.threshold, "inference_ms_per_1000_rows": ms,
        "n_parameters": model.n_parameters}, indent=2, default=float))
    print(f"\nSaved to {models_dir}/ and {reports}/lstm_report.json  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
