"""Replay the production model's predictions through the alert engine and measure the effect.

  * main table (TEST period): configured engine vs two naive alternatives
  * sensitivity grid (VALIDATION period): how confirm_steps / cooldown trade alert volume against lead time
    (parameters in config were fixed a priori and are NOT tuned on the test period)

    python scripts/simulate_alerts.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alerts.engine import AlertEngine  # noqa: E402
from src.config import get_config, resolve_path  # noqa: E402
from src.features.dataset import build_dataset  # noqa: E402
from src.prediction.predictor import CongestionPredictor  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

STEP_MIN = 5
EVENT_MIN_STEPS, GRACE_STEPS = 3, 3          # same event definition as the model evaluation


def events_from_truth(df: pd.DataFrame) -> list[tuple[str, pd.Timestamp]]:
    """(site, prediction-row time of onset) for runs of >= 3 congested TARGET rows."""
    out = []
    for site, g in df.sort_values("prediction_time").groupby("site_id"):
        y = g["target_binary"].to_numpy()
        t = g["prediction_time"].to_numpy()
        edges = np.diff(np.r_[0, y, 0])
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if e - s >= EVENT_MIN_STEPS:
                out.append((site, pd.Timestamp(t[s])))
    return out


def run_engine(df: pd.DataFrame, cfg: dict) -> tuple[dict, dict[str, list], dict[str, int]]:
    """Return (notification counts, alert intervals per site, notifications per site excluding recovery notices)."""
    eng = AlertEngine(cfg)
    counts = {"RAISED": 0, "ESCALATED": 0, "REMINDER": 0, "DEESCALATED": 0, "RECOVERED": 0}
    open_at, intervals, last_t, per_site = {}, {}, {}, {}
    d = df.sort_values(["prediction_time", "site_id"])
    for site, t, lvl, p, st in zip(d.site_id, d.prediction_time, d.risk_level, d.congestion_probability, d.predicted_state):
        last_t[site] = t
        for a in eng.evaluate(site, t, lvl, p, st):
            counts[a.kind] += 1
            if a.kind != "RECOVERED":
                per_site[site] = per_site.get(site, 0) + 1
            if a.kind == "RAISED":
                open_at[site] = t
            elif a.kind == "RECOVERED":
                intervals.setdefault(site, []).append((open_at.pop(site), t))
    for site, t0 in open_at.items():
        intervals.setdefault(site, []).append((t0, last_t[site]))
    return counts, intervals, per_site


def raw_intervals(df: pd.DataFrame) -> dict[str, list]:
    """Runs of consecutive risky (>= MEDIUM) predictions, as if every positive were an alert."""
    out = {}
    for site, g in df.sort_values("prediction_time").groupby("site_id"):
        f = (g["risk_level"] != "LOW").to_numpy()
        t = g["prediction_time"].to_numpy()
        edges = np.diff(np.r_[0, f.astype(int), 0])
        out[site] = [(pd.Timestamp(t[s]), pd.Timestamp(t[e - 1])) for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    return out


def score(intervals: dict, events: list, df: pd.DataFrame, horizon_min: float) -> dict:
    grace = pd.Timedelta(minutes=STEP_MIN * GRACE_STEPS)
    leads, covered = [], 0
    for site, t0 in events:
        hit = [(a, b) for a, b in intervals.get(site, []) if a <= t0 + grace and b >= t0]
        if hit:
            covered += 1
            leads.append((t0 + pd.Timedelta(minutes=horizon_min) - min(a for a, _ in hit)).total_seconds() / 60)
    false = 0
    for site, g in df.groupby("site_id"):
        tt, y = g["prediction_time"].to_numpy(), g["target_binary"].to_numpy()
        for a, b in intervals.get(site, []):
            m = (tt >= np.datetime64(a)) & (tt <= np.datetime64(b))
            false += int(not y[m].any())
    site_days = len(df) * STEP_MIN / 1440
    return {"event_coverage": covered / max(len(events), 1), "n_events": len(events),
            "median_lead_min": float(np.median(leads)) if leads else float("nan"),
            "false_alert_episodes_per_site_day": false / site_days, "alert_episodes": sum(len(v) for v in intervals.values())}


def main() -> None:
    cfg = get_config()
    setup_logging(cfg)
    clean_path = resolve_path(cfg, "processed_dir") / cfg["preprocessing"]["clean_filename"]
    if not clean_path.exists():
        sys.exit("Run generate_data.py and prepare_data.py first.")
    pred = CongestionPredictor.load(cfg)
    clean = pd.read_csv(clean_path, parse_dates=["timestamp"])
    b = build_dataset(clean, cfg)
    frame = pred.predict_frame(clean)
    truth = lambda d: d[["site_id", "timestamp", "target_binary"]].rename(columns={"timestamp": "prediction_time"})  # noqa: E731
    parts = {n: frame.merge(truth(getattr(b, n)), on=["site_id", "prediction_time"]) for n in ("val", "test")}
    H = cfg["prediction"]["horizon_minutes"]
    print(f"Model: {pred.name} | alert config: confirm {cfg['alerts']['confirm_steps']}, cooldown {cfg['alerts']['cooldown_minutes']} min, "
          f"recovery {cfg['alerts']['recovery_steps']} | SIMULATED DATA\n")

    # ---------------- main table on TEST ----------------
    df = parts["test"]
    events = events_from_truth(df)
    site_days = len(df) * STEP_MIN / 1440
    rows = []
    risky = int((df.risk_level != "LOW").sum())
    r = score(raw_intervals(df), events, df, H)
    rows.append({"scenario": "A. notify on EVERY risky prediction", "notifications_per_site_day": risky / site_days, **r})
    nd = copy.deepcopy(cfg); nd["alerts"].update(confirm_steps=1, cooldown_minutes=0, recovery_steps=1, rearm_minutes=0, send_reminders=False)
    counts, iv, _ = run_engine(df, nd)
    rows.append({"scenario": "B. notify when risk starts (no debounce, no hysteresis)",
                 "notifications_per_site_day": sum(counts.values()) / site_days, **score(iv, events, df, H)})
    counts, iv, per_site = run_engine(df, cfg)
    rows.append({"scenario": "C. CONFIGURED alert engine", "notifications_per_site_day": sum(counts.values()) / site_days,
                 "pages_excl_recovery_per_site_day": (sum(counts.values()) - counts["RECOVERED"]) / site_days, **score(iv, events, df, H)})
    tab = pd.DataFrame(rows)
    print(f"=== TEST period: {len(df):,} predictions, {site_days:.0f} site-days, {len(events)} congestion events ===")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Configured engine, notifications by kind:", counts)
    cells = clean.groupby("site_id")[["region", "location"]].first()
    top = sorted(per_site.items(), key=lambda kv: -kv[1])[:5]
    print("\nBusiest cells (notifications excluding recovery notices) on the test period:")
    for site, n in top:
        print(f"  {site}  {cells.region[site]:<18s} {cells.location[site]:<22s} {n:>3d}  ({n / (len(df) / df.site_id.nunique() * STEP_MIN / 1440):.1f} per day)")
    share_core = sum(n for s_, n in per_site.items() if cells.region[s_] == "urban_core") / max(sum(per_site.values()), 1)
    print(f"  share of all notifications from urban_core cells: {share_core:.0%} ({int((cells.region == 'urban_core').sum())} of {len(cells)} cells)")

    # ---------------- sensitivity on VALIDATION ----------------
    dv = parts["val"]; ev = events_from_truth(dv); sdv = len(dv) * STEP_MIN / 1440
    grid = []
    for cs in (1, 2, 3):
        for cd in (30, 60, 120):
            c = copy.deepcopy(cfg); c["alerts"].update(confirm_steps=cs, cooldown_minutes=cd)
            cnt, iv, _ = run_engine(dv, c)
            grid.append({"confirm_steps": cs, "cooldown_min": cd, "notifications_per_site_day": sum(cnt.values()) / sdv, **score(iv, ev, dv, H)})
    g = pd.DataFrame(grid)
    print("\n=== Sensitivity on VALIDATION (tune here, not on test) ===")
    print(g[["confirm_steps", "cooldown_min", "notifications_per_site_day", "event_coverage", "median_lead_min",
             "false_alert_episodes_per_site_day"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = resolve_path(cfg, "reports_dir") / "alert_simulation.json"
    out.write_text(json.dumps({"test": tab.to_dict(orient="records"), "test_kinds": counts, "test_busiest_cells": [{"site_id": k, "region": cells.region[k], "location": cells.location[k], "notifications": v} for k, v in top],
                               "validation_sensitivity": g.to_dict(orient="records"),
                               "data_note": "SIMULATED data"}, indent=2, default=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
