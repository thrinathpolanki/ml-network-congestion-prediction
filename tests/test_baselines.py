import numpy as np
import pandas as pd
import pytest

from src.models.baselines import MajorityClass, Persistence, SeasonalNaive, ThresholdDetector

STEP = pd.Timedelta("5min")


def _train(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    return pd.DataFrame({"bandwidth_utilization_pct": x,
                         "target_state_id": np.digitize(x, [60, 75, 90])})


def test_majority_predicts_mode_with_constant_score():
    tr = pd.DataFrame({"target_state_id": [0, 0, 0, 1, 2]})
    m = MajorityClass().fit(tr)
    assert (m.predict(tr) == 0).all() and np.ptp(m.congestion_score(tr)) == 0


def test_threshold_recovers_known_cutoffs():
    tr = _train()
    m = ThresholdDetector().fit(tr)
    assert np.all(np.diff(m.cuts_) > 0)
    assert (m.predict(tr) == tr["target_state_id"]).mean() > 0.97
    np.testing.assert_allclose(m.cuts_, [60, 75, 90], atol=2.0)


def test_threshold_handles_nan_and_range():
    m = ThresholdDetector().fit(_train())
    p = m.predict(pd.DataFrame({"bandwidth_utilization_pct": [np.nan, 10.0, 99.0]}))
    assert set(p) <= {0, 1, 2, 3} and p[2] == 3


def test_persistence_returns_current_state():
    df = pd.DataFrame({"state_now": [0.0, 2.0, np.nan, 3.0], "score_now": [0.1, 0.5, np.nan, 0.9]})
    assert list(Persistence().predict(df)) == [0, 2, 0, 3]


def _hist(days=4):
    ts = pd.date_range("2026-01-01", periods=days * 288, freq="5min")
    return pd.DataFrame({"site_id": "A", "timestamp": ts,
                         "state_now": (ts.hour // 6).astype(float), "score_now": ts.hour / 24.0})


def test_seasonal_naive_reads_same_time_yesterday():
    h = _hist()
    m = SeasonalNaive(1, 6, STEP).set_history(h)
    df = h[h.timestamp >= "2026-01-03"]
    expected = ((df.timestamp + STEP * 6).dt.hour // 6).to_numpy()   # pattern repeats daily
    assert (m.predict(df) == expected).all()


def test_seasonal_naive_falls_back_to_persistence_without_history():
    h = _hist()
    m = SeasonalNaive(1, 6, STEP).set_history(h)
    first = h.head(10)                                   # t+H-1d precedes the data
    assert (m.predict(first) == first["state_now"].astype(int)).all()


def test_seasonal_naive_cannot_peek_forward():
    with pytest.raises(ValueError):
        SeasonalNaive(0, 6, STEP)                        # period 0 < horizon
    h = _hist()
    m = SeasonalNaive(1, 6, STEP).set_history(h)
    row = h[h.timestamp == "2026-01-03 12:00"]
    before = m.predict(row)
    h2 = h.copy()
    h2.loc[h2.timestamp > "2026-01-03 12:00", "state_now"] = 3.0      # corrupt everything later
    assert (SeasonalNaive(1, 6, STEP).set_history(h2).predict(row) == before).all()


def test_seasonal_requires_history():
    with pytest.raises(RuntimeError):
        SeasonalNaive(1, 6, STEP).predict(_hist().head(5))
