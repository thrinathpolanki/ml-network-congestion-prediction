"""Risk-engine sanity report.

  A. What does each risk level actually mean? Observed congestion frequency per level (validation, test).
     Descriptive only: the thresholds are configuration and are NOT tuned on these numbers.
  B. Serving-path parity: raw, dirty data through the ONLINE history preparation vs the OFFLINE-cleaned
     training path, for the same model.

    python scripts/risk_report.py        # needs data/raw + data/processed + trained models
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, resolve_path  # noqa: E402
from src.features.dataset import build_dataset  # noqa: E402
from src.models.evaluate import calibration_table  # noqa: E402
from src.prediction.predictor import CongestionPredictor  # noqa: E402
from src.prediction.risk import RISK_LEVELS  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

KEYS = ["site_id", "timestamp"]


def main() -> None:
    cfg = get_config()
    setup_logging(cfg)
    raw_path = resolve_path(cfg, "raw_dir") / cfg["data"]["raw_filename"]
    clean_path = resolve_path(cfg, "processed_dir") / cfg["preprocessing"]["clean_filename"]
    if not (raw_path.exists() and clean_path.exists()):
        sys.exit("Run generate_data.py and prepare_data.py first.")
    pred = CongestionPredictor.load(cfg)
    print(f"Model: {pred.name} | tau {pred.model.threshold:.3f} | risk thresholds {pred.risk.thresholds}\n")
    clean = pd.read_csv(clean_path, parse_dates=["timestamp"])
    bundle = build_dataset(clean, cfg)
    truth = pd.concat([bundle.val.assign(split="val"), bundle.test.assign(split="test")])[KEYS + ["target_binary", "split"]]

    off = pred.predict_frame(clean).rename(columns={"prediction_time": "timestamp"}).merge(truth, on=KEYS, how="inner")

    print("A. Observed congestion frequency by risk level (descriptive; thresholds not tuned)")
    rows = []
    for split, g in off.groupby("split"):
        for lvl in RISK_LEVELS:
            s = g[g.risk_level == lvl]
            rows.append({"split": split, "risk_level": lvl, "rows": len(s),
                         "mean_predicted_prob": float(s.congestion_probability.mean()) if len(s) else np.nan,
                         "observed_congestion_rate": float(s.target_binary.mean()) if len(s) else np.nan})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    cal = {}
    for split, g in off.groupby("split"):
        _, st = calibration_table(g.target_binary.to_numpy(), g.congestion_probability.to_numpy())
        cal[split] = st
        print(f"  calibration on {split}: ECE {st['ece']:.4f}, Brier {st['brier']:.4f}"
              + ("  (isotonic was fitted on validation: in-sample)" if split == "val" else "  (out-of-sample)"))

    print("\nB. Serving-path parity: raw dirty data through the ONLINE preparation vs offline-cleaned data")
    raw = pd.read_csv(raw_path, parse_dates=["timestamp"])
    on = pred.predict_frame(raw).rename(columns={"prediction_time": "timestamp"})
    m = off[off.split == "test"].merge(on, on=KEYS, suffixes=("_off", "_on"))
    d = (m.congestion_probability_off - m.congestion_probability_on).abs()
    agree_bin = ((m.congestion_probability_off >= pred.model.threshold) == (m.congestion_probability_on >= pred.model.threshold)).mean()
    out = {"rows_compared": int(len(m)), "mean_abs_prob_diff": float(d.mean()), "p99_abs_prob_diff": float(d.quantile(0.99)),
           "max_abs_prob_diff": float(d.max()), "share_rows_diff_gt_0.01": float((d > 0.01).mean()),
           "share_rows_diff_gt_0.05": float((d > 0.05).mean()),
           "congested_decision_agreement": float(agree_bin),
           "risk_level_agreement": float((m.risk_level_off == m.risk_level_on).mean()),
           "pr_auc_offline_path": float(average_precision_score(m.target_binary, m.congestion_probability_off)),
           "pr_auc_online_path": float(average_precision_score(m.target_binary, m.congestion_probability_on))}
    for k, v in out.items():
        print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")
    big = m[d > 0.3]
    print(f"  rows with |diff| > 0.3: {len(big)} ({len(big) / len(m):.2%})")

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (split, col) in enumerate((("val", "#7f8c8d"), ("test", "#c0392b"))):
        t = tab[tab.split == split]
        ax.bar(np.arange(4) + i * 0.38, t.observed_congestion_rate * 100, 0.36, color=col, label=split)
        for x, r, n in zip(np.arange(4) + i * 0.38, t.observed_congestion_rate * 100, t.rows):
            ax.text(x, (0 if np.isnan(r) else r) + 1, f"n={n:,}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(np.arange(4) + 0.19); ax.set_xticklabels(RISK_LEVELS); ax.set_ylabel("congested 30 min later (%)")
    ax.set_title("What each risk level means in practice (simulated data)"); ax.legend(); ax.set_ylim(0, 115)
    fig.tight_layout(); fig_dir = resolve_path(cfg, "reports_dir") / "figures"; fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "20_risk_levels_observed.png", dpi=110); plt.close(fig)
    (resolve_path(cfg, "reports_dir") / "risk_report.json").write_text(json.dumps(
        {"model": pred.name, "levels": tab.to_dict(orient="records"), "calibration": cal, "serving_parity": out},
        indent=2, default=float))


if __name__ == "__main__":
    main()
