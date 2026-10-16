"""Generate docs/MODEL_COMPARISON.md from the saved comparison outputs (no hand-copied numbers).

    python scripts/make_comparison_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402


def pm(mu: float, sd: float, nd: int = 3) -> str:
    return f"{mu:.{nd}f}" + (f" ± {sd:.{nd}f}" if sd and sd > 0 else "")


def main() -> None:
    cfg = get_config()
    rep = ROOT / cfg["paths"]["reports_dir"]
    sel = json.loads((rep / "selection_val.json").read_text())
    res = json.loads((rep / "test_results_summary.json").read_text())
    tab = {r["model"]: r for r in res["test_table"]}
    L = ["# Model comparison", "",
         "> **SIMULATED DATA.** One generated dataset; seeds vary model initialisation/sampling only. "
         "Nothing here is real telecom performance.", "",
         "## Process (fixed before looking at test results)", "",
         f"1. Every model trained with {len(res['seeds'])} seeds {res['seeds']}; early stopping on validation.",
         "2. Operating points (calibration + F2 threshold) fitted on **validation** for every model alike.",
         f"3. Selection rule on validation only: keep **{sel['decision']['default']}** unless a challenger beats it by "
         f">= {sel['decision']['z_min']} pooled seed-SDs on PR-AUC or on *operator area* (event recall averaged over "
         f"0.2-1.0 false alerts per site-day) without being clearly worse on the other. Needs >= 3 seeds.",
         f"4. Decision written to `models/production_model.json` at {sel['decided_at_utc']} UTC, **before** the test set was scored.",
         "5. Test set scored once (lock file `reports/test_results.json`).", "",
         "## Validation evidence used for selection", "",
         "| Model | PR-AUC (mean ± sd) | Operator area (mean ± sd) |", "|---|---|---|"]
    for n, v in sel["val_summary"].items():
        L.append(f"| {n} | {pm(v['pr_auc']['mean'], v['pr_auc']['sd'], 4)} | {pm(v['operator_area']['mean'], v['operator_area']['sd'])} |")
    L += ["", "z-scores vs default (positive = challenger better):", ""]
    for n, z in sel["decision"]["comparisons"].items():
        L.append(f"- {n}: PR-AUC z = {z['pr_auc']:.2f}, operator-area z = {z['operator_area']:.2f}")
    L += ["", f"**Chosen by the rule: `{sel['decision']['chosen']}`.** {sel['decision']['reason']}", "",
          "## Test results (mean ± sd over seeds; baselines are deterministic)", "",
          "| Model | Accuracy | Macro-F1 | PR-AUC | Cong. recall | Cong. precision | New-onset recall | Event recall | False alerts / site-day |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res["test_table"]:
        L.append("| " + " | ".join([r["model"]] + [pm(r[m], r[m + "_sd"]) for m in
                 ["accuracy", "macro_f1", "pr_auc", "cong_recall", "cong_precision", "recall_new_onset",
                  "event_recall", "false_alerts_per_site_day"]]) + " |")
    L += ["", "95% CIs (day-block bootstrap, first seed): " + "; ".join(
        f"{r['model']}: macro-F1 {r['macro_f1_ci95_seed1']}, recall {r['cong_recall_ci95_seed1']}"
        for r in res["test_table"] if "macro_f1_ci95_seed1" in r and isinstance(r.get('macro_f1_ci95_seed1'), str)
        and "tau" in r["model"]), ""]
    o = res["overlap"]
    L += ["## Do the two models make the same mistakes? (tuned tau, first seed)", "",
          f"- Congested rows: {o['n_congested']}. Missed by XGBoost: {o['missed_by_a']}; by LSTM: {o['missed_by_b']}; by both: {o['missed_by_both']} "
          f"(Jaccard of misses {o['miss_jaccard']:.2f}).",
          f"- False-alarm rows: XGBoost {o['false_alarm_a']}, LSTM {o['false_alarm_b']}, both {o['false_alarm_both']} "
          f"(Jaccard {o['false_alarm_jaccard']:.2f}).",
          f"- Caught by XGBoost {o['caught_by_a']}, by LSTM {o['caught_by_b']}, by either {o['caught_by_either']}.", "",
          "## Cost, complexity and practicality", "",
          "| | XGBoost | LSTM |", "|---|---|---|"]
    c, lat = res["complexity"], res["latency_ms"]
    L += [f"| Size | {c['xgboost']['model_file_kb']:.0f} KB, {c['xgboost']['trees_total']} trees | {c['lstm']['parameters']:,} parameters |",
          f"| Inputs | {c['xgboost']['features']} engineered features | {c['lstm']['inputs_per_step']} per step x seq + {c['lstm']['static_inputs']} static |",
          f"| History needed | {c['xgboost']['history_steps_needed']} readings | {c['lstm']['history_steps_needed']} readings |",
          f"| Training time (mean, this machine) | {c['xgboost']['train_seconds_mean']:.0f} s | {c['lstm']['train_seconds_mean']:.0f} s |",
          f"| End-to-end latency per site request (median / p95) | {lat['xgboost']['median_ms']:.0f} / {lat['xgboost']['p95_ms']:.0f} ms | {lat['lstm']['median_ms']:.0f} / {lat['lstm']['p95_ms']:.0f} ms |",
          "| Interpretability | Built-in gain + TreeSHAP | None built in (post-hoc methods only) |",
          "| Near-real-time suitability | Yes | Yes |", "",
          f"Both are dominated by feature construction (pandas), not by the model. Hybrid (average of both) would take "
          f"{lat['hybrid']['median_ms']:.0f} ms median.", ""]
    crit = res["critical"]
    L += ["## Known weaknesses", "",
          f"- CRITICAL class (tuned tau, first seed): XGBoost recall {crit['XGBoost']['recall']:.2f} / precision {crit['XGBoost']['precision']:.2f}; "
          f"LSTM recall {crit['LSTM']['recall']:.2f} / precision {crit['LSTM']['precision']:.2f} (support {crit['XGBoost']['support']}).",
          "- Per-region results for regions with almost no congestion (rural, suburban) are not informative.",
          "- The hybrid was " + ("scored on test." if res["hybrid_evaluated_on_test"] else "NOT scored on test, because validation did not select it."),
          "- The selection rule compares each challenger only with the default, not with each other, and uses single-criterion z-scores; "
          "with 4 seeds the SD estimates are crude. See the narrative in the project report.", ""]
    out = ROOT / "docs" / "MODEL_COMPARISON.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
