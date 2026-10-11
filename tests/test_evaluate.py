import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (align_frames, bootstrap_ci, build_frame, compare_frames,
                                 evaluate_frame, event_detection, plot_confusions)

STEP = pd.Timedelta("5min")


def mk(y_true, y_pred, state_now=None, site="A", p=None, start="2026-01-01"):
    n = len(y_true)
    yp = np.asarray(y_pred)
    return pd.DataFrame({
        "site_id": site, "timestamp": pd.date_range(start, periods=n, freq="5min"),
        "y_true": y_true, "y_pred": y_pred,
        "p_congested": (yp >= 2).astype(float) if p is None else p,
        "state_now": [0] * n if state_now is None else state_now})


YT = [0] * 6 + [1] * 3 + [2] * 5 + [0] * 4 + [3] * 4 + [0] * 3        # all 4 classes present


def test_perfect_predictions(cfg):
    m = evaluate_frame(mk(YT, YT), cfg)
    assert m["accuracy"] == m["macro_f1"] == m["cong_recall"] == m["cong_precision"] == 1.0
    assert m["event_recall"] == 1.0 and m["false_alert_episodes_per_site_day"] == 0.0
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0


def test_all_normal_predictions(cfg):
    m = evaluate_frame(mk(YT, [0] * len(YT)), cfg)
    assert m["cong_recall"] == 0.0 and m["event_recall"] == 0.0 and m["recall_critical"] == 0.0
    assert m["accuracy"] == pytest.approx(13 / len(YT))
    assert m["per_class"]["NORMAL"]["recall"] == 1.0


def test_constant_score_gives_auc_half_and_pr_equals_prevalence(cfg):
    f = mk(YT, [0] * len(YT), p=np.zeros(len(YT)))
    m = evaluate_frame(f, cfg)
    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["pr_auc"] == pytest.approx(m["prevalence"])


def test_event_detection_grace_window():
    yt = [0] * 5 + [2] * 5 + [0] * 10                       # event = rows 5..9
    early = [0] * 6 + [2] + [0] * 13                        # flagged on 2nd event row -> detected
    late = [0] * 9 + [2] + [0] * 10                         # flagged on 5th event row -> missed
    assert event_detection(mk(yt, early), STEP, 3, 3)["event_recall"] == 1.0
    assert event_detection(mk(yt, late), STEP, 3, 3)["event_recall"] == 0.0


def test_false_alert_episodes_counted_per_site_day():
    yt = [0] * 5 + [2] * 5 + [0] * 10
    pred = [0] * 6 + [2] + [0] * 8 + [2, 2] + [0] * 3       # one true-ish alert, one false (rows 15-16)
    r = event_detection(mk(yt, pred), STEP, 3, 3)
    assert r["false_alert_episodes_per_site_day"] == pytest.approx(1 / (20 * 5 / 1440))


def test_short_flicker_events_ignored():
    yt = [0] * 5 + [2, 2] + [0] * 5                         # 2 steps < min_len 3
    r = event_detection(mk(yt, yt), STEP, 3, 3)
    assert r["n_events"] == 0 and np.isnan(r["event_recall"])


def test_events_do_not_merge_across_sites_or_gaps():
    a = mk([0] * 4 + [2] * 4, [2] * 8, site="A")
    b = mk([2] * 4 + [0] * 4, [2] * 8, site="B")
    assert event_detection(pd.concat([a, b]), STEP, 3, 3)["n_events"] == 2
    g = mk([2] * 8, [2] * 8)
    g = g.drop(index=[4])                                    # a time gap splits the run (4 | 3 rows)
    assert event_detection(g, STEP, 3, 3)["n_events"] == 2


def test_new_onset_vs_continuing_recall(cfg):
    yt = [0] * 4 + [2] * 6 + [0] * 4 + [2] * 6
    now = [0] * 4 + [0] + [2] * 5 + [0] * 4 + [2] * 6        # first event new; second continuing
    pred = [0] * 4 + [0] * 6 + [0] * 4 + [2] * 6              # model only catches continuing one
    m = evaluate_frame(mk(yt, pred, state_now=now), cfg)
    assert m["n_new_onset"] == 1 and m["recall_new_onset"] == 0.0
    assert m["n_continuing"] == 11 and m["recall_continuing"] == pytest.approx(6 / 11)


def test_align_frames_uses_common_rows_only():
    a, b = mk(YT, YT), mk(YT, YT).iloc[5:]
    out = align_frames({"a": a, "b": b})
    assert len(out["a"]) == len(out["b"]) == len(b)
    pd.testing.assert_frame_equal(out["a"][["site_id", "timestamp"]], out["b"][["site_id", "timestamp"]])


def _random_frame(n_days=12, seed=0):
    rng = np.random.default_rng(seed)
    n = n_days * 288
    yt = rng.choice(4, n, p=[.8, .1, .07, .03])
    yp = np.where(rng.random(n) < 0.7, yt, rng.choice(4, n))
    return mk(yt, yp, start="2026-01-01", p=(yp >= 2) + rng.random(n) * 0.1)


def test_bootstrap_is_reproducible_and_brackets_estimate(cfg):
    f = _random_frame()
    c1, c2 = bootstrap_ci(f, cfg), bootstrap_ci(f, cfg)
    assert c1 == c2
    m = evaluate_frame(f, cfg)
    lo, hi = c1["macro_f1"]
    assert lo <= m["macro_f1"] <= hi and hi - lo > 0


def test_compare_frames_end_to_end_and_plot(cfg, tmp_path):
    f, g = _random_frame(seed=1), _random_frame(seed=2)
    table, results, _ = compare_frames({"m1": f, "m2": g.iloc[100:]}, cfg)
    assert set(table["model"]) == {"m1", "m2"} and results["m1"]["n_rows"] == results["m2"]["n_rows"]
    assert "macro_f1_ci95" in table.columns
    assert plot_confusions(results, tmp_path / "cm.png") and (tmp_path / "cm.png").exists()


def test_build_frame_shapes():
    df = pd.DataFrame({"site_id": "A", "timestamp": pd.date_range("2026-01-01", periods=3, freq="5min"),
                       "target_state_id": [0, 1, 2], "state_now": [0.0, 1.0, 2.0]})
    f = build_frame(df, [0, 1, 2], [0.1, 0.2, 0.9], probs=np.eye(4)[[0, 1, 2]])
    assert list(f.columns[:6]) == ["site_id", "timestamp", "y_true", "state_now", "y_pred", "p_congested"]
    assert f["y_true"].dtype.kind == "i" and "p_class3" in f.columns
