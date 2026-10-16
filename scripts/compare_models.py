"""Phase 10: formal model comparison (baselines vs XGBoost vs LSTM [vs hybrid]).

    python scripts/compare_models.py --quick   # 1 seed, short training: checks the pipeline (no test lock)
    python scripts/compare_models.py           # full run: several seeds, scores the TEST set ONCE

Process
  A. train every model for every seed (train period; early stopping on validation).
     Each finished seed is cached in reports/compare_cache, so an interrupted run resumes where it stopped.
  B. VALIDATION ONLY: fit operating points, apply the pre-declared selection rule, write the decision to disk
  C. only now score the TEST set (a lock file prevents a second, tuned-by-peeking run)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, horizon_steps, resolve_path  # noqa: E402
from src.features.dataset import build_dataset  # noqa: E402
from src.models.baselines import (MajorityClass, Persistence, SeasonalNaive, ThresholdDetector)  # noqa: E402
from src.models.compare import (OperatingPoint, align_probs, error_overlap, operator_area,  # noqa: E402
                                operator_curve, select_model, summarize, time_requests)
from src.models.evaluate import build_frame, compare_frames, plot_confusions  # noqa: E402
from src.models.lstm_model import LSTMCongestionModel, SequenceBuilder, column_sets  # noqa: E402
from src.models.xgb_model import XGBCongestionModel  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

KEYS = ["site_id", "timestamp"]
METRICS = ["accuracy", "macro_f1", "cong_precision", "cong_recall", "cong_f1", "pr_auc", "roc_auc",
           "recall_critical", "recall_new_onset", "event_recall", "false_alerts_per_site_day"]
LABEL = {"xgboost": "XGBoost", "lstm": "LSTM", "hybrid": "Hybrid (avg)"}
COLOR = {"xgboost": "#2c3e50", "lstm": "#c0392b", "hybrid": "#2e9e5b"}


def meta_of(df: pd.DataFrame) -> pd.DataFrame:
    return df[KEYS + ["target_state_id", "state_now"]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-score the test set despite the lock file")
    ap.add_argument("--fresh", action="store_true", help="ignore cached per-seed training results and retrain everything")
    a = ap.parse_args()
    cfg = get_config()
    cc = cfg["comparison"]
    if a.quick:
        cc["seeds"] = [42]
        cfg["xgboost"]["n_estimators"] = 40
        cfg["lstm"].update(max_epochs=3, windows_per_epoch=20000)
    setup_logging(cfg)
    models_dir, reports = resolve_path(cfg, "models_dir"), resolve_path(cfg, "reports_dir")
    out = reports / ("quick" if a.quick else "")
    out.mkdir(exist_ok=True)
    fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)
    pred_dir = out / "predictions"; pred_dir.mkdir(exist_ok=True)
    lock = reports / cc["test_lock_file"]
    if not a.quick and lock.exists() and not a.force:
        sys.exit(f"{lock} exists: the test set was already scored. Re-scoring after seeing results invalidates the "
                 f"comparison. Use --force only for a documented re-run.")
    clean_path = resolve_path(cfg, "processed_dir") / cfg["preprocessing"]["clean_filename"]
    if not clean_path.exists():
        sys.exit("Run scripts/prepare_data.py first.")
    xmeta_path = models_dir / cfg["xgboost"]["meta_filename"]
    if not xmeta_path.exists():
        sys.exit("Run scripts/train_xgboost.py first (its tuned parameters are reused).")
    xgb_params = json.loads(xmeta_path.read_text())["params"]

    t0 = time.time()
    clean = pd.read_csv(clean_path, parse_dates=["timestamp"])
    bundle = build_dataset(clean, cfg)
    builder = bundle.builder
    feats_all = builder.transform(clean, drop_incomplete=False)
    dyn, static = column_sets(builder)
    t1 = pd.Timestamp(bundle.report["baselines_fitted_on_rows_before"])
    sb = SequenceBuilder(dyn, static, cfg["lstm"]["seq_len"]).fit(feats_all[feats_all.timestamp < t1])
    tr_s, va_s, te_s = (sb.build(feats_all, d) for d in (bundle.train, bundle.val, bundle.test))
    cols, seeds, taus = builder.feature_columns, cc["seeds"], cc["taus"]
    print(f"Data ready ({time.time() - t0:.0f}s). Seeds: {seeds}. Training {len(seeds) * 2} models...")

    # ---------------- A. train (resumable) ----------------
    P: dict[tuple[str, int], dict] = {}
    train_info: dict[tuple[str, int], dict] = {}
    models_seed42 = {}
    cache_dir = out / "compare_cache"; cache_dir.mkdir(exist_ok=True)
    # a cache entry is only valid for exactly this data, these models' settings and these parameters
    fingerprint = hashlib.sha256(json.dumps({
        "xgb_params": xgb_params, "xgboost": cfg["xgboost"], "lstm": cfg["lstm"], "features": cfg["features"],
        "labeling": cfg["labeling"], "rows": [len(bundle.train), len(bundle.val), len(bundle.test)],
        "latency_sum": round(float(bundle.train["latency_ms"].sum()), 3), "n_features": len(cols)},
        sort_keys=True, default=str).encode()).hexdigest()[:16]
    for s in seeds:
        c = copy.deepcopy(cfg); c["project"]["seed"] = s
        cpath, mdir = cache_dir / f"seed{s}.joblib", cache_dir / f"models_seed{s}"
        cached = None
        if cpath.exists() and not a.fresh:
            try:
                cached = joblib.load(cpath)
            except Exception:  # noqa: BLE001 - a corrupt/partial cache file just means: retrain
                cached = None
            if cached is not None and cached.get("fingerprint") != fingerprint:
                cached = None
        if cached is not None:
            P.update(cached["P"]); train_info.update(cached["train_info"])
            if s == seeds[0]:
                models_seed42 = {"xgboost": XGBCongestionModel.load(mdir, c), "lstm": LSTMCongestionModel.load(mdir, c)[0]}
            print(f"  seed {s} loaded from cache ({time.time() - t0:.0f}s)")
        else:
            set_global_seed(s)
            xm = XGBCongestionModel(c).fit(bundle.train, bundle.val, cols, xgb_params)
            P[("xgboost", s)] = {"val": (meta_of(bundle.val), xm.predict_proba(bundle.val)),
                                 "test": (meta_of(bundle.test), xm.predict_proba(bundle.test))}
            train_info[("xgboost", s)] = {"train_seconds": xm.meta["fit_seconds"], "trees": xm.meta["best_iteration"] * 4}
            lm = LSTMCongestionModel(c, dyn, static)
            lm.fit(tr_s, va_s)
            P[("lstm", s)] = {"val": (va_s.meta, lm.predict_proba(va_s)), "test": (te_s.meta, lm.predict_proba(te_s))}
            train_info[("lstm", s)] = {"train_seconds": lm.meta["train_seconds"], "epochs": lm.meta["epochs_run"],
                                       "params": lm.n_parameters}
            if s == seeds[0]:
                xm.save(mdir); lm.save(mdir, sb)
                models_seed42 = {"xgboost": xm, "lstm": lm}
            tmp = cpath.with_suffix(".tmp")                       # write-then-rename: a kill mid-write cannot leave a half file
            joblib.dump({"fingerprint": fingerprint, "P": {k: v for k, v in P.items() if k[1] == s},
                         "train_info": {k: v for k, v in train_info.items() if k[1] == s}}, tmp)
            tmp.replace(cpath)
            print(f"  seed {s} done ({time.time() - t0:.0f}s)")
        for sp in ("val", "test"):                       # hybrid = mean of the two class-probability vectors
            m, px, pl = align_probs(P[("xgboost", s)][sp][0], P[("xgboost", s)][sp][1],
                                    P[("lstm", s)][sp][0], P[("lstm", s)][sp][1])
            P.setdefault(("hybrid", s), {})[sp] = (m, 0.5 * (px + pl))

    # ---------------- B. validation only: operating points + selection ----------------
    thr = cfg["lstm"]["threshold"]
    ops, val_summary, val_detail = {}, {n: {"pr_auc": [], "operator_area": []} for n in LABEL}, []
    for (name, s), d in P.items():
        meta, p = d["val"]
        op = OperatingPoint.fit(p, meta["target_state_id"].to_numpy(), thr)
        ops[(name, s)] = op
        curve = operator_curve(op, p, meta, cfg, taus)
        pr = float(average_precision_score(meta["target_state_id"].to_numpy() >= 2, p[:, 2:].sum(1)))
        area = operator_area(curve, tuple(cc["operator_fa_range"]))
        val_summary[name]["pr_auc"].append(pr); val_summary[name]["operator_area"].append(area)
        val_detail.append({"model": name, "seed": s, "val_pr_auc": pr, "val_operator_area": area, "tau": op.tau})
    decision = select_model(val_summary, cc["default_model"], float(cc["z_min"]))
    sel = {"decided_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "basis": "validation only",
           "decision": decision, "per_seed": val_detail,
           "val_summary": {n: {k: summarize(v) for k, v in m.items()} for n, m in val_summary.items()}}
    (out / "selection_val.json").write_text(json.dumps(sel, indent=2, default=float))
    print("\n=== VALIDATION EVIDENCE & SELECTION (decided BEFORE scoring test) ===")
    for n in LABEL:
        v = sel["val_summary"][n]
        print(f"  {LABEL[n]:13s} PR-AUC {v['pr_auc']['mean']:.4f} ± {v['pr_auc']['sd']:.4f} | "
              f"operator-area {v['operator_area']['mean']:.3f} ± {v['operator_area']['sd']:.3f}")
    print("  z-scores vs", cc["default_model"], decision["comparisons"])
    print(f"  CHOSEN: {decision['chosen']} - {decision['reason']}")
    if not a.quick:
        (models_dir / "production_model.json").write_text(json.dumps({
            "model": decision["chosen"], "reason": decision["reason"], "selected_on": "validation",
            "decided_before_test_scoring": True, "decided_at_utc": sel["decided_at_utc"], "seeds": seeds,
            "data_note": "Selected on SIMULATED data."}, indent=2))

    # ---------------- C. score the TEST set once ----------------
    print("\nScoring TEST set...")
    evaluated = ["xgboost", "lstm"] + (["hybrid"] if decision["chosen"] == "hybrid" else [])
    hist = pd.concat([bundle.train, bundle.val, bundle.test])[KEYS + ["state_now", "score_now"]]
    hs, step = horizon_steps(cfg), pd.Timedelta(cfg["data"]["sampling_freq"])
    base = [MajorityClass(), ThresholdDetector(cfg["baselines"]["threshold_feature"]), Persistence()]
    base += [SeasonalNaive(int(k), hs, step).set_history(hist) for k in cfg["baselines"]["seasonal_days"]]
    base_frames = {}
    for b in base:
        b.fit(bundle.train)
        base_frames[b.name] = build_frame(bundle.test, b.predict(bundle.test), b.congestion_score(bundle.test))
    per_seed: dict[str, dict[str, list[float]]] = {}
    tables, test_results, curves = {}, {}, {}
    for s in seeds:
        frames = dict(base_frames)
        for n in evaluated:
            meta, p = P[(n, s)]["test"]
            frames[f"{LABEL[n]} (argmax)"] = build_frame(meta, p.argmax(1), p[:, 2:].sum(1), p)
            frames[f"{LABEL[n]} (tuned tau)"] = build_frame(meta, ops[(n, s)].decide(p), p[:, 2:].sum(1), p)
            curves[(n, s)] = operator_curve(ops[(n, s)], p, meta, cfg, taus)
            np.savez_compressed(pred_dir / f"test__{n}_seed{s}.npz", p=p, keys=meta[KEYS].astype(str).to_numpy())
        table, results, aligned = compare_frames(frames, cfg, with_ci=(s == seeds[0]))
        tables[s], test_results[s] = table, results
        for name, r in results.items():
            for m in METRICS:
                key = "false_alert_episodes_per_site_day" if m == "false_alerts_per_site_day" else m
                per_seed.setdefault(name, {}).setdefault(m, []).append(r[key])
        if s == seeds[0]:
            aligned0, results0 = aligned, results
    rows = []
    for name, d in per_seed.items():
        row = {"model": name}
        for m in METRICS:
            sm = summarize(d[m]); row[m] = sm["mean"]; row[m + "_sd"] = sm["sd"]
        if name in test_results[seeds[0]] and "ci95" in test_results[seeds[0]][name]:
            ci = test_results[seeds[0]][name]["ci95"]
            row["macro_f1_ci95_seed1"] = f"[{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}]"
            row["cong_recall_ci95_seed1"] = f"[{ci['cong_recall'][0]:.3f}, {ci['cong_recall'][1]:.3f}]"
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg.to_csv(out / "model_comparison_test.csv", index=False)
    show = ["model", "macro_f1", "pr_auc", "cong_recall", "cong_precision", "recall_new_onset", "event_recall", "false_alerts_per_site_day"]
    print(f"\n=== TEST ({results0['Persistence (state now)']['n_rows']:,} rows; SIMULATED DATA; mean over {len(seeds)} seeds) ===")
    fmt = agg[show].copy()
    for m in show[1:]:
        fmt[m] = [f"{mu:.3f}" + (f" ± {sd:.3f}" if sd > 0 else "") for mu, sd in zip(agg[m], agg[m + "_sd"])]
    print(fmt.to_string(index=False))
    print("\n95% CIs of the first-seed models (day-block bootstrap):")
    print(agg.dropna(subset=["macro_f1_ci95_seed1"])[["model", "macro_f1_ci95_seed1", "cong_recall_ci95_seed1"]].to_string(index=False))

    # ---- diagnostics (seed 1): overlap, per-region, per-class ----
    s0 = seeds[0]
    ax_, al_ = aligned0["XGBoost (tuned tau)"], aligned0["LSTM (tuned tau)"]
    overlap = error_overlap(ax_["y_true"].to_numpy(), ax_["y_pred"].to_numpy(), al_["y_pred"].to_numpy())
    reg = bundle.test[KEYS + ["region"]]
    region_rows = []
    for lab, f in (("XGBoost", ax_), ("LSTM", al_)):
        j = f.merge(reg, on=KEYS)
        for r, g in j.groupby("region"):
            yb, pb = g.y_true >= 2, g.y_pred >= 2
            region_rows.append({"model": lab, "region": r, "rows": len(g), "congested_rows": int(yb.sum()),
                                "recall": float((pb & yb).sum() / yb.sum()) if yb.sum() else float("nan"),
                                "precision": float((pb & yb).sum() / pb.sum()) if pb.sum() else float("nan")})
    region_df = pd.DataFrame(region_rows)
    print("\nError overlap (tuned tau, seed 1):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in overlap.items()})
    print("\nPer-region (tuned tau, seed 1):")
    print(region_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    crit = {n: results0[f"{n} (tuned tau)"]["per_class"]["CRITICAL"] for n in ("XGBoost", "LSTM")}
    print("\nCRITICAL class (tuned tau, seed 1):", {k: {a_: round(b_, 3) for a_, b_ in v.items()} for k, v in crit.items()})

    # ---- end-to-end latency (single-site request: history -> features -> probabilities) ----
    rng = np.random.default_rng(0)
    by_site = {k: g.sort_values("timestamp").reset_index(drop=True) for k, g in clean.groupby("site_id")}
    reqs = []
    for _, r in bundle.test.sample(min(len(bundle.test), cc["latency_requests"]), random_state=0).iterrows():
        g = by_site[r.site_id]; pos = int(g.index[g.timestamp == r.timestamp][0])
        reqs.append(g.iloc[pos - builder.required_history_steps + 1: pos + 1])
    xm, lm = models_seed42["xgboost"], models_seed42["lstm"]
    f_x = lambda i: xm.predict_proba(builder.transform_latest(reqs[i % len(reqs)]))  # noqa: E731
    f_l = lambda i: lm.proba_from_arrays(*sb.window_from_features(  # noqa: E731
        builder.transform(reqs[i % len(reqs)].tail(sb.seq_len), drop_incomplete=False)))
    f_h = lambda i: (f_x(i), f_l(i))  # noqa: E731
    lat = {"xgboost": time_requests(f_x, cc["latency_requests"]), "lstm": time_requests(f_l, cc["latency_requests"]),
           "hybrid": time_requests(f_h, cc["latency_requests"])}
    print("\nEnd-to-end latency per single-site request (ms, median / p95):",
          {k: f"{v['median_ms']:.1f} / {v['p95_ms']:.1f}" for k, v in lat.items()})
    complexity = {
        "xgboost": {"model_file_kb": round((models_dir / cfg["xgboost"]["model_filename"]).stat().st_size / 1024, 1),
                    "trees_total": train_info[("xgboost", s0)]["trees"], "features": len(cols),
                    "history_steps_needed": builder.required_history_steps,
                    "train_seconds_mean": float(np.mean([train_info[("xgboost", s)]["train_seconds"] for s in seeds]))},
        "lstm": {"parameters": train_info[("lstm", s0)]["params"], "inputs_per_step": len(dyn), "static_inputs": len(static),
                 "history_steps_needed": sb.seq_len,
                 "train_seconds_mean": float(np.mean([train_info[("lstm", s)]["train_seconds"] for s in seeds]))}}
    print("Complexity:", complexity)

    # ---- figures ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    for n in evaluated:
        for s in seeds:
            meta, p = P[(n, s)]["test"]; yb = meta["target_state_id"].to_numpy() >= 2
            pr, rc, _ = precision_recall_curve(yb, p[:, 2:].sum(1))
            ax[0].plot(rc, pr, color=COLOR[n], alpha=0.5, lw=1, label=LABEL[n] if s == seeds[0] else None)
            c = curves[(n, s)]
            ax[1].plot(c.false_alerts_per_site_day, c.event_recall, "o-", color=COLOR[n], alpha=0.55, ms=3, lw=1,
                       label=LABEL[n] if s == seeds[0] else None)
    for name, f in base_frames.items():
        if name.startswith("Majority"):
            continue
        r = results0[name]
        ax[0].scatter(r["cong_recall"], r["cong_precision"], s=50, zorder=3, label=name)
        ax[1].scatter(r["false_alert_episodes_per_site_day"], r["event_recall"], s=50, zorder=3, label=name)
    ax[0].set_xlabel("recall (congested)"); ax[0].set_ylabel("precision"); ax[0].set_title("TEST: precision-recall (one line per seed)")
    ax[1].set_xlabel("false alert episodes per site-day"); ax[1].set_ylabel("event recall"); ax[1].set_title("TEST: operator trade-off (tau sweep)")
    ax[0].legend(fontsize=7, loc="lower left"); ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(fig_dir / "17_test_pr_and_operator.png", dpi=110); plt.close(fig)
    plot_confusions({k: results0[k] for k in ("Persistence (state now)", "XGBoost (tuned tau)", "LSTM (tuned tau)")},
                    fig_dir / "18_test_confusions.png", "TEST, first seed (simulated data)")
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for a_, (m, lab, sfx) in zip(ax, [("pr_auc", "PR-AUC", "tuned tau"), ("macro_f1", "macro-F1 (argmax)", "argmax"),
                                       ("event_recall", "event recall (tau)", "tuned tau"),
                                       ("false_alerts_per_site_day", "false alerts / site-day (tau)", "tuned tau")]):
        for i, n in enumerate(evaluated):
            v = per_seed[f"{LABEL[n]} ({sfx})"][m]
            a_.scatter(np.full(len(v), i) + rng.uniform(-.08, .08, len(v)), v, color=COLOR[n], s=40)
        a_.set_xticks(range(len(evaluated))); a_.set_xticklabels([LABEL[n] for n in evaluated]); a_.set_title(lab)
    fig.suptitle(f"Seed-to-seed spread on TEST ({len(seeds)} seeds, same simulated dataset)")
    fig.tight_layout(); fig.savefig(fig_dir / "19_seed_variability.png", dpi=110); plt.close(fig)

    summary = {"seeds": seeds, "selection": decision, "test_table": agg.to_dict(orient="records"),
               "overlap": overlap, "per_region": region_df.to_dict(orient="records"), "critical": crit,
               "latency_ms": lat, "complexity": complexity, "evaluated_models": evaluated,
               "hybrid_evaluated_on_test": "hybrid" in evaluated,
               "data_note": "SIMULATED data; one generated dataset; seeds vary model initialisation/sampling only."}
    (out / "test_results_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if not a.quick:
        lock.write_text(json.dumps({"scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "seeds": seeds, "chosen_before_scoring": decision["chosen"]}, indent=2))
    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {out}")


if __name__ == "__main__":
    main()
