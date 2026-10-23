import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.config import horizon_steps
from src.data.generator import generate_dataset
from src.features.dataset import build_dataset
from src.features.engineering import InsufficientHistoryError
from src.models.lstm_model import LSTMCongestionModel, SequenceBuilder, column_sets
from src.models.xgb_model import XGBCongestionModel
from src.prediction.history import prepare_history
from src.prediction.predictor import CongestionPredictor, ModelConfigError
from src.prediction.risk import RISK_LEVELS, RiskEngine, horizon_label


# ----------------------------- risk engine --------------------------------- #
def test_levels_follow_thresholds_and_are_monotone(cfg):
    r = RiskEngine(cfg, model_tau=0.17)
    assert r.thresholds == [0.17, 0.40, 0.70]
    assert [r.level(p) for p in (0.0, 0.169, 0.17, 0.39, 0.40, 0.699, 0.70, 1.0)] == \
        ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH", "CRITICAL", "CRITICAL"]
    ids = r.level_ids(np.linspace(0, 1, 200))
    assert (np.diff(ids) >= 0).all()


def test_thresholds_are_configurable_and_validated(cfg):
    c = copy.deepcopy(cfg)
    c["risk"]["levels"] = {"medium": 0.3, "high": 0.5, "critical": 0.9}
    assert RiskEngine(c, 0.17).thresholds == [0.3, 0.5, 0.9]
    c["risk"]["levels"] = {"medium": 0.6, "high": 0.5, "critical": 0.9}
    with pytest.raises(ValueError):
        RiskEngine(c, 0.17)
    c = copy.deepcopy(cfg); del c["risk"]["actions"]["HIGH"]
    with pytest.raises(ValueError):
        RiskEngine(c, 0.17)


def test_assessment_fields_and_semantics(cfg):
    r = RiskEngine(cfg, 0.17)
    probs = np.array([0.1, 0.1, 0.5, 0.3])
    a = r.assess("CELL_014", pd.Timestamp("2026-02-01 18:00"), 0.87, probs, 2, "lstm", {"x": 1})
    d = a.to_dict()
    assert d["risk_level"] == "CRITICAL" and d["predicted_state"] == "CONGESTED"
    assert d["confidence"] == 0.87 and d["prediction_horizon"] == "30_minutes"
    assert d["target_time"] == "2026-02-01T18:30:00" and set(d["class_probabilities"]) == {"NORMAL", "WARNING", "CONGESTED", "CRITICAL"}
    low = r.assess("A", pd.Timestamp("2026-02-01 03:00"), 0.04, np.array([.9, .06, .03, .01]), 0)
    assert low.risk_level == "LOW" and low.confidence == pytest.approx(0.96)       # confidence = P(predicted side)
    assert json.dumps(d)                                                           # JSON-serialisable
    with pytest.raises(ValueError):
        r.assess("A", pd.Timestamp("2026-02-01"), 1.2, probs, 2)


def test_horizon_label_follows_config(cfg):
    c = copy.deepcopy(cfg); c["prediction"]["horizon_minutes"] = 60
    r = RiskEngine(c, 0.2)
    a = r.assess("A", pd.Timestamp("2026-02-01 10:00"), 0.5, np.array([.2, .2, .3, .3]), 2)
    assert a.prediction_horizon == "60_minutes" and a.target_time == "2026-02-01T11:00:00"
    assert horizon_label(30) == "30_minutes"


# ----------------------------- history preparation ------------------------- #
@pytest.fixture(scope="module")
def raw(cfg):
    return generate_dataset(cfg, n_sites=2, start="2026-01-05", end="2026-01-08", seed=2, inject_issues=False)


def test_prepare_history_identity_on_clean_regular_data(cfg, raw):
    out = prepare_history(raw.sample(frac=1.0, random_state=0), cfg)       # also order-independent
    pd.testing.assert_frame_equal(out.drop(columns=["data_source"]).reset_index(drop=True),
                                  raw.sort_values(["site_id", "timestamp"]).drop(columns=["data_source"]).reset_index(drop=True),
                                  check_dtype=False)


def test_prepare_history_dedupes_sanitizes_and_fills_short_gaps_only(cfg, raw):
    g = raw[raw.site_id == "CELL_001"].reset_index(drop=True)
    dup = pd.concat([g, g.iloc[[10]].assign(latency_ms=999.0)])           # newest duplicate wins
    bad = dup.copy()
    bad.loc[bad.index[20], "download_throughput_mbps"] = -5.0              # invalid -> NaN -> ffill
    bad.loc[bad.index[30:33], "latency_ms"] = np.nan                       # 3-step gap -> filled
    bad.loc[bad.index[100:110], "jitter_ms"] = np.nan                      # 10-step gap -> stays NaN
    out = prepare_history(bad, cfg)
    assert not out.duplicated(["site_id", "timestamp"]).any() and len(out) == len(g)
    assert out.loc[10, "latency_ms"] == 999.0
    assert out["download_throughput_mbps"].iloc[20] == g["download_throughput_mbps"].iloc[19]
    assert out["latency_ms"].iloc[30:33].notna().all()
    assert out["jitter_ms"].iloc[100:110].isna().all()


def test_prepare_history_rebuilds_missing_rows_and_rejects_empty(cfg, raw):
    g = raw[raw.site_id == "CELL_001"].reset_index(drop=True)
    out = prepare_history(g.drop(index=[40, 41]), cfg)
    assert len(out) == len(g) and out["latency_ms"].iloc[40] == g["latency_ms"].iloc[39]
    with pytest.raises(ValueError):
        prepare_history(g.assign(timestamp="not a date"), cfg)


# ----------------------------- predictor ----------------------------------- #
@pytest.mark.parametrize("name", ["xgboost", "lstm"])
def test_predict_site_returns_valid_assessment(served, name):
    c, d, data = served
    p = CongestionPredictor.load(c, d, model=name)
    site = data[data.site_id == "CELL_001"].sort_values("timestamp")
    cut = pd.Timestamp("2026-01-20 19:00")
    a = p.predict_site(site[site.timestamp <= cut].tail(p.required_steps + 10))
    assert a.site_id == "CELL_001" and a.prediction_time == cut.isoformat()
    assert pd.Timestamp(a.target_time) - pd.Timestamp(a.prediction_time) == pd.Timedelta(minutes=30)
    assert 0 <= a.congestion_probability <= 1 and a.risk_level in RISK_LEVELS and a.model == name
    assert a.context["region"] and a.recommended_action


@pytest.mark.parametrize("name", ["xgboost", "lstm"])
def test_single_site_matches_backtest_frame(served, name):
    """Serving path (last N readings) must equal the batch/backtest path at the same timestamp."""
    c, d, data = served
    p = CongestionPredictor.load(c, d, model=name)
    frame = p.predict_frame(data)
    site = data[data.site_id == "CELL_002"].sort_values("timestamp")
    for ts in ("2026-01-12 08:00", "2026-01-18 20:30", "2026-01-24 12:00"):
        a = p.predict_site(site[site.timestamp <= ts].tail(p.required_steps))
        row = frame[(frame.site_id == "CELL_002") & (frame.prediction_time == pd.Timestamp(ts))].iloc[0]
        assert a.congestion_probability == pytest.approx(row.congestion_probability, abs=2e-3)
        assert a.risk_level == row.risk_level and a.predicted_state == row.predicted_state


def test_predict_many_isolates_bad_sites(served):
    c, d, data = served
    p = CongestionPredictor.load(c, d, model="lstm")
    ok = data[data.site_id == "CELL_001"].tail(60)
    short = data[data.site_id == "CELL_002"].tail(5)
    results, errors = p.predict_many(pd.concat([ok, short]))
    assert [r.site_id for r in results] == ["CELL_001"] and "CELL_002" in errors


def test_insufficient_history_and_multi_site_rejected(served):
    c, d, data = served
    p = CongestionPredictor.load(c, d, model="xgboost")
    with pytest.raises(InsufficientHistoryError):
        p.predict_site(data[data.site_id == "CELL_001"].head(20))
    with pytest.raises(ValueError):
        p.predict_site(data.groupby("site_id").tail(100))


def test_invalid_latest_reading_is_survivable(served):
    c, d, data = served
    p = CongestionPredictor.load(c, d, model="lstm")
    h = data[data.site_id == "CELL_001"].sort_values("timestamp").head(200).copy()
    good = p.predict_site(h)
    h.iloc[-1, h.columns.get_loc("latency_ms")] = -50.0                     # impossible -> NaN -> carried forward
    assert p.predict_site(h).risk_level in RISK_LEVELS and good.site_id == "CELL_001"


def test_auto_model_selection_and_overrides(served):
    c, d, _ = served
    assert CongestionPredictor.load(c, d).name == "lstm"                    # production_model.json
    assert CongestionPredictor.load(c, d, model="xgboost").name == "xgboost"
    (d / "production_model.json").write_text(json.dumps({"model": "hybrid"}))
    with pytest.raises(FileNotFoundError, match="fit_hybrid"):          # hybrid needs its own fitted operating point
        CongestionPredictor.load(c, d)
    (d / "production_model.json").write_text(json.dumps({"model": "lstm"}))
    with pytest.raises(ValueError):
        CongestionPredictor.load(c, d, model="svm")


def test_horizon_mismatch_is_rejected(served):
    c, d, _ = served
    c2 = copy.deepcopy(c); c2["prediction"]["horizon_minutes"] = 60
    with pytest.raises(ModelConfigError):
        CongestionPredictor.load(c2, d, model="xgboost")
    assert horizon_steps(c2) == 12


def test_lstm_loads_with_architecture_from_checkpoint_not_config(served):
    c, d, data = served
    c2 = copy.deepcopy(c); c2["lstm"]["hidden_size"] = 128                  # someone edited the config after training
    p = CongestionPredictor.load(c2, d, model="lstm")
    assert p.predict_site(data[data.site_id == "CELL_001"].head(60)).risk_level in RISK_LEVELS


def test_missing_artifacts_and_info(served, tmp_path):
    c, d, _ = served
    with pytest.raises(FileNotFoundError):
        CongestionPredictor.load(c, tmp_path, model="xgboost")
    info = CongestionPredictor.load(c, d, model="lstm").info()
    assert info["horizon_minutes"] == 30 and info["history_required_steps"] == 36 and "SIMULATED" in info["data_note"]


# ----------------------------- updated data: cell position travels with the assessment ----------------- #
def test_assessment_context_carries_the_cell_position_when_available(served):
    c, d, data = served
    p = CongestionPredictor.load(c, d, model="lstm")
    w = data[data.site_id == "CELL_001"].sort_values("timestamp").iloc[:60]
    a = p.predict_site(w)
    ctx = a.context
    assert ctx["latitude"] == pytest.approx(w["latitude"].iloc[-1]) and ctx["longitude"] == pytest.approx(w["longitude"].iloc[-1])
    assert ctx["location"] == f"{ctx['latitude']:.6f},{ctx['longitude']:.6f}"
    assert json.dumps(a.to_dict())
    bare = p.predict_site(w.drop(columns=["latitude", "longitude", "location"]))      # a source without positions still works
    assert "latitude" not in bare.context and "location" not in bare.context
    assert bare.congestion_probability == a.congestion_probability                    # position never affects the prediction
