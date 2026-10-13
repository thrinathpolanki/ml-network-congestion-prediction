"""CLI: tune, train, calibrate and evaluate XGBoost on the VALIDATION period (test is untouched).

    python scripts/train_xgboost.py              # with hyper-parameter search
    python scripts/train_xgboost.py --no-tune    # base parameters only (faster)
Needs: python scripts/build_features.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, horizon_steps, resolve_path  # noqa: E402
from src.features.engineering import FeatureBuilder  # noqa: E402
from src.models.baselines import (KEYS, MajorityClass, Persistence, SeasonalNaive,  # noqa: E402
                                  ThresholdDetector)
from src.models.evaluate import (build_frame, calibration_table, compare_frames,  # noqa: E402
                                 evaluate_frame, plot_confusions, time_predict)
from src.models.xgb_model import XGBCongestionModel  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

SHOW = ["model", "accuracy", "macro_f1", "cong_precision", "cong_recall", "pr_auc",
        "recall_new_onset", "event_recall", "false_alerts_per_site_day"]


def figures(model, val, p, pc, baseline_frames, tab, fig_dir):
    yb = val["target_state_id"].to_numpy() >= 2
    # 11 - precision/recall and operator trade-off
    taus = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep = []
    for t in taus:
        r = evaluate_frame(build_frame(val, model.decide(p, t), p[:, 2:].sum(1)), model.cfg)
        sweep.append((t, r["cong_precision"], r["cong_recall"], r["event_recall"], r["false_alert_episodes_per_site_day"]))
    sweep = pd.DataFrame(sweep, columns=["tau", "precision", "recall", "event_recall", "false_alerts"])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    pr, rc, _ = precision_recall_curve(yb, p[:, 2:].sum(1))
    ax[0].plot(rc, pr, lw=2, color="#c0392b", label="XGBoost (all thresholds)")
    ax[1].plot(sweep["false_alerts"], sweep["event_recall"], "o-", color="#c0392b", label="XGBoost (tau sweep)")
    for tt, fa, er in zip(sweep["tau"], sweep["false_alerts"], sweep["event_recall"]):
        ax[1].annotate(f"{tt:g}", (fa, er), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for name, f in baseline_frames.items():
        if name.startswith(("Majority", "XGBoost")):
            continue
        r = evaluate_frame(f, model.cfg)
        ax[0].scatter(r["cong_recall"], r["cong_precision"], s=60, label=name, zorder=3)
        ax[1].scatter(r["false_alert_episodes_per_site_day"], r["event_recall"], s=60, label=name, zorder=3)
    r = evaluate_frame(baseline_frames["XGBoost (tuned tau)"], model.cfg)
    ax[0].scatter(r["cong_recall"], r["cong_precision"], marker="*", s=220, color="black", label=f"XGBoost chosen tau={model.threshold:.2f}", zorder=4)
    ax[1].scatter(r["false_alert_episodes_per_site_day"], r["event_recall"], marker="*", s=220, color="black", zorder=4)
    ax[0].axhline(yb.mean(), ls=":", color="grey"); ax[0].set_xlabel("recall (congested)"); ax[0].set_ylabel("precision")
    ax[0].set_title("Binary congestion, VALIDATION (dotted = chance)"); ax[0].legend(fontsize=7, loc="lower left"); ax[0].set_ylim(0, 1.02)
    ax[1].set_xlabel("false alert episodes per site-day"); ax[1].set_ylabel("event recall")
    ax[1].set_title("Operator view: events caught vs false alerts (labels = tau)"); ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(fig_dir / "11_xgb_precision_recall_val.png", dpi=110); plt.close(fig)
    # 12 - importance
    imp, shap = model.importance(), model.shap_top(val)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    g = imp.groupby("group")["gain_share"].sum().sort_values()
    ax[0].barh(g.index, g.values * 100, color="#2c3e50"); ax[0].set_xlabel("% of total gain"); ax[0].set_title("Importance by feature group")
    ax[1].barh(shap["feature"][::-1], shap["mean_abs_shap"][::-1], color="#c0392b")
    ax[1].set_title("Top 20 features: mean |SHAP| towards Congested+Critical"); ax[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "12_xgb_importance.png", dpi=110); plt.close(fig)
    # 13 - calibration
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for lab, pp, c in (("raw", p[:, 2:].sum(1), "#e67e22"), ("calibrated", pc, "#2e9e5b")):
        t, st = calibration_table(yb, pp)
        ax.plot(t["mean_pred"], t["frac_pos"], "o-", color=c, label=f"{lab} (ECE {st['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("predicted P(congested)"); ax.set_ylabel("observed frequency")
    ax.set_title("Reliability on validation (in-sample for isotonic)"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "13_xgb_calibration_val.png", dpi=110); plt.close(fig)
    return imp, shap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--figures-only", action="store_true", help="reload the saved model; skip training")
    a = ap.parse_args()
    cfg = get_config()
    setup_logging(cfg)
    set_global_seed(cfg["project"]["seed"])
    d, pre = resolve_path(cfg, "processed_dir"), cfg["features"]["processed_prefix"]
    models_dir, reports = resolve_path(cfg, "models_dir"), resolve_path(cfg, "reports_dir")
    fig_dir = reports / "figures"; fig_dir.mkdir(exist_ok=True)
    try:
        train, val, test = (pd.read_parquet(d / f"{pre}_{s}.parquet") for s in ("train", "val", "test"))
        builder = FeatureBuilder.load(models_dir / cfg["features"]["builder_filename"])
    except FileNotFoundError:
        sys.exit("Feature files/builder missing. Run: python scripts/build_features.py")
    cols = builder.feature_columns
    group_of = {c: g for g, v in builder.feature_groups.items() for c in v}

    model = XGBCongestionModel(cfg)
    trials = None
    best = {}
    if a.figures_only:
        model = XGBCongestionModel.load(models_dir, cfg)
        print("Loaded saved model; skipping training.")
    elif cfg["xgboost"]["tuning"]["enabled"] and not a.no_tune:
        print(f"Hyper-parameter search ({cfg['xgboost']['tuning']['n_iter']} trials, objective = validation PR-AUC)...")
        best, trials = model.tune(train, val, cols)
        trials.to_csv(reports / "xgb_tuning_trials.csv", index=False)
        print(trials[["trial", "max_depth", "learning_rate", "min_child_weight", "class_weight_power",
                      "val_pr_auc", "best_iteration", "seconds"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not a.figures_only:
        print("\nTraining final model on the full training period...")
        model.feature_groups = group_of
        model.fit(train, val, cols, best)
        model.calibrate(val)
    tab = model.fit_threshold(val)

    p = model.predict_proba(val)
    pc = model.congestion_probability(val)
    raw = p[:, 2:].sum(1)
    hs, step = horizon_steps(cfg), pd.Timedelta(cfg["data"]["sampling_freq"])
    hist = pd.concat([train, val, test])[KEYS + ["state_now", "score_now"]]
    base = [MajorityClass(), ThresholdDetector(cfg["baselines"]["threshold_feature"]), Persistence()]
    base += [SeasonalNaive(int(k), hs, step).set_history(hist) for k in cfg["baselines"]["seasonal_days"]]
    frames = {}
    for b in base:
        b.fit(train)
        frames[b.name] = build_frame(val, b.predict(val), b.congestion_score(val))
    frames["XGBoost (argmax)"] = build_frame(val, p.argmax(1), raw, p)
    frames["XGBoost (tuned tau)"] = build_frame(val, model.decide(p), raw, p)
    table, results, _ = compare_frames(frames, cfg)

    print(f"\n=== VALIDATION ({len(val):,} rows; SIMULATED DATA; val also used for tuning/threshold => optimistic) ===")
    print(table[SHOW].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(table[["model", "macro_f1_ci95", "cong_recall_ci95"]].to_string(index=False))

    print("\nOperating points (calibrated P(cong) >= tau):")
    ops = []
    for t in (0.10, 0.20, 0.30, 0.40, 0.50, 0.70):
        r = evaluate_frame(build_frame(val, model.decide(p, t), raw, p), cfg)
        ops.append({"tau": t, "precision": r["cong_precision"], "recall": r["cong_recall"],
                    "new_onset_recall": r["recall_new_onset"], "event_recall": r["event_recall"],
                    "false_alerts_per_site_day": r["false_alert_episodes_per_site_day"]})
    ops = pd.DataFrame(ops)
    print(ops.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nChosen tau = {model.threshold:.2f} (maximises F{model.xc['threshold']['beta']:.0f} on validation)")

    yb = val["target_state_id"].to_numpy() >= 2
    _, cal_raw = calibration_table(yb, raw)
    _, cal_cal = calibration_table(yb, pc)
    print(f"Calibration  raw: ECE {cal_raw['ece']:.4f} Brier {cal_raw['brier']:.4f} | "
          f"calibrated: ECE {cal_cal['ece']:.4f} Brier {cal_cal['brier']:.4f}")
    ms = time_predict(model.predict, val)
    print(f"Inference: {ms:.1f} ms per 1,000 rows (feature matrix already built)")

    imp, shap = figures(model, val, p, pc, frames, tab, fig_dir)
    plot_confusions({k: results[k] for k in ("Persistence (state now)", "XGBoost (argmax)", "XGBoost (tuned tau)")},
                    fig_dir / "14_xgb_confusions_val.png", "Validation (simulated data)")
    print("\nTop 10 features by mean |SHAP|:", ", ".join(shap["feature"].head(10)))
    print("Gain share by group (%):", {k: round(v * 100, 1) for k, v in
                                       imp.groupby("group")["gain_share"].sum().sort_values(ascending=False).items()})

    if not a.figures_only:
        model.save(models_dir, extra_meta={
            "train_period": [str(train.timestamp.min()), str(train.timestamp.max())],
            "val_period": [str(val.timestamp.min()), str(val.timestamp.max())],
            "val_metrics_note": "validation metrics are optimistic: validation was used for early stopping, search, calibration, threshold",
            "inference_ms_per_1000_rows": round(ms, 2), "tuned": bool(best)})
    report = {"best_params": model.params, "threshold": model.threshold, "best_iteration": model.meta["best_iteration"],
              "val_results": results, "operating_points": ops.to_dict(orient="records"),
              "calibration": {"raw": cal_raw, "calibrated": cal_cal},
              "top_shap": shap.to_dict(orient="records"),
              "group_gain_share": imp.groupby("group")["gain_share"].sum().to_dict(),
              "inference_ms_per_1000_rows": ms}
    (reports / "xgb_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved model + meta to {models_dir}/ and report to {reports}/xgb_report.json")


if __name__ == "__main__":
    main()
