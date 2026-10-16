import numpy as np
import pandas as pd
import pytest

from src.models.compare import (OperatingPoint, align_probs, error_overlap, operator_area, pooled_z,
                                select_model, summarize, time_requests)

THR = {"beta": 2.0, "grid_min": 0.02, "grid_max": 0.98, "grid_points": 97}


def _probs(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(4, n, p=[.8, .1, .07, .03])
    logit = np.eye(4)[y] * 2.0 + rng.normal(0, 1, (n, 4))
    p = np.exp(logit) / np.exp(logit).sum(1, keepdims=True)
    return p, y


def test_operating_point_threshold_and_decisions():
    p, y = _probs()
    op = OperatingPoint.fit(p, y, THR)
    assert 0.02 <= op.tau <= 0.98
    d = op.decide(p)
    assert set(np.unique(d)) <= {0, 1, 2, 3}
    prob = op.prob(p)
    assert prob.min() >= 0 and prob.max() <= 1
    rec = [(op.decide(p, t) >= 2)[y >= 2].mean() for t in (0.1, 0.3, 0.6, 0.9)]
    assert all(a >= b for a, b in zip(rec, rec[1:]))


def test_beta_two_picks_more_sensitive_threshold_than_beta_half():
    p, y = _probs(seed=1)
    lo = OperatingPoint.fit(p, y, {**THR, "beta": 2.0}).tau
    hi = OperatingPoint.fit(p, y, {**THR, "beta": 0.5}).tau
    assert lo <= hi


def test_operator_area_known_curves():
    good = pd.DataFrame({"false_alerts_per_site_day": [0.0, 0.2, 1.0], "event_recall": [0.5, 0.9, 0.9]})
    bad = pd.DataFrame({"false_alerts_per_site_day": [0.0, 1.0], "event_recall": [0.0, 0.5]})
    assert operator_area(good, (0.2, 1.0)) == pytest.approx(0.9)
    assert operator_area(bad, (0.2, 1.0)) == pytest.approx(0.3, abs=1e-6)       # mean of 0.1..0.5
    assert operator_area(good, (0.2, 1.0)) > operator_area(bad, (0.2, 1.0))


def test_pooled_z_behaviour():
    assert pooled_z([0.8, 0.8001, 0.7999], [0.9, 0.9001, 0.8999]) > 10
    assert abs(pooled_z([0.80, 0.82, 0.78], [0.81, 0.79, 0.80])) < 1
    assert pooled_z([0.8, 0.8], [0.8, 0.8]) == 0.0                              # floor avoids division by zero


def _val(x_pr, l_pr, x_area, l_area):
    return {"xgboost": {"pr_auc": x_pr, "operator_area": x_area}, "lstm": {"pr_auc": l_pr, "operator_area": l_area}}


def test_selection_keeps_default_on_tie():
    d = select_model(_val([.815, .817, .813], [.814, .812, .816], [.7, .72, .68], [.71, .69, .70]), "xgboost", 2.0)
    assert d["chosen"] == "xgboost" and "simplest" in d["reason"]


def test_selection_adopts_clearly_better_challenger():
    d = select_model(_val([.815, .816, .814], [.84, .841, .839], [.7, .71, .69], [.70, .71, .69]), "xgboost", 2.0)
    assert d["chosen"] == "lstm" and d["comparisons"]["lstm"]["pr_auc"] > 2


def test_selection_rejects_challenger_that_is_clearly_worse_elsewhere():
    d = select_model(_val([.815, .816, .814], [.84, .841, .839], [.80, .81, .79], [.60, .61, .59]), "xgboost", 2.0)
    assert d["chosen"] == "xgboost"


def test_error_overlap_counts():
    y = np.array([0, 2, 2, 3, 0, 0])
    a = np.array([0, 2, 0, 0, 2, 0])
    b = np.array([0, 0, 0, 3, 2, 2])
    o = error_overlap(y, a, b)
    assert o["n_congested"] == 3 and o["missed_by_a"] == 2 and o["missed_by_b"] == 2 and o["missed_by_both"] == 1
    assert o["false_alarm_both"] == 1 and o["caught_by_either"] == 2
    assert o["miss_jaccard"] == pytest.approx(1 / 3)


def test_time_requests_and_summarize():
    t = time_requests(lambda i: sum(range(1000)), n=20)
    assert t["n"] == 20 and 0 <= t["median_ms"] <= t["p95_ms"]
    s = summarize([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["sd"] == pytest.approx(1.0) and s["n"] == 3


def test_align_probs_keeps_common_rows_in_order():
    ts = pd.date_range("2026-01-01", periods=6, freq="5min")
    a = pd.DataFrame({"site_id": "A", "timestamp": ts})
    b = pd.DataFrame({"site_id": "A", "timestamp": ts[2:]})
    pa, pb = np.arange(6)[:, None] * np.ones((1, 4)), (np.arange(4) + 100)[:, None] * np.ones((1, 4))
    meta, x, y = align_probs(a, pa, b, pb)
    assert len(meta) == 4 and x[:, 0].tolist() == [2, 3, 4, 5] and y[:, 0].tolist() == [100, 101, 102, 103]


def test_selection_needs_at_least_three_seeds():
    d = select_model(_val([.8], [.9], [.7], [.9]), "xgboost", 2.0)
    assert d["chosen"] == "xgboost" and "seed" in d["reason"]
