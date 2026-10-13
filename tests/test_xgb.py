import copy

import numpy as np
import pandas as pd
import pytest

from src.data.generator import generate_dataset
from src.features.dataset import build_dataset
from src.models.evaluate import calibration_table
from src.models.xgb_model import XGBCongestionModel, sample_weights, select_threshold


@pytest.fixture(scope="module")
def small_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["xgboost"]["n_estimators"] = 30
    c["xgboost"]["early_stopping_rounds"] = 5
    c["xgboost"]["base_params"].update(max_depth=4, colsample_bytree=0.5)
    c["xgboost"]["tuning"].update(n_iter=2, train_stride=4)
    return c


@pytest.fixture(scope="module")
def bundle(cfg):
    d = generate_dataset(cfg, n_sites=4, start="2026-01-05", end="2026-01-26", seed=5,
                         intensity=1.3, inject_issues=False)
    return build_dataset(d, cfg)


@pytest.fixture(scope="module")
def fitted(small_cfg, bundle):
    cols = bundle.builder.feature_columns
    m = XGBCongestionModel(small_cfg).fit(bundle.train, bundle.val, cols)
    m.feature_groups = {c: g for g, v in bundle.builder.feature_groups.items() for c in v}
    m.calibrate(bundle.val)
    m.fit_threshold(bundle.val)
    return m


def test_sample_weights_balance_classes():
    y = np.array([0] * 90 + [1] * 5 + [2] * 4 + [3] * 1)
    w1, w0, wh = sample_weights(y, 1.0), sample_weights(y, 0.0), sample_weights(y, 0.5)
    sums = [w1[y == k].sum() for k in range(4)]
    assert np.allclose(sums, sums[0])                       # fully balanced
    assert (w0 == 1).all()
    assert w0[y == 3][0] < wh[y == 3][0] < w1[y == 3][0]    # tempered in between


def test_proba_shapes_and_ranges(fitted, bundle):
    p = fitted.predict_proba(bundle.val)
    assert p.shape == (len(bundle.val), 4) and np.allclose(p.sum(1), 1, atol=1e-4)
    for use_thr in (True, False):
        s = fitted.predict(bundle.val, use_threshold=use_thr)
        assert set(np.unique(s)) <= {0, 1, 2, 3}
    cp = fitted.congestion_probability(bundle.val)
    assert cp.min() >= 0 and cp.max() <= 1


def test_model_learns_something_on_validation(fitted, bundle):
    from sklearn.metrics import average_precision_score
    yb = bundle.val["target_state_id"].to_numpy() >= 2
    ap = average_precision_score(yb, fitted.raw_congestion_score(bundle.val))
    assert ap > 3 * yb.mean()                               # far above the chance level (prevalence)


def test_lower_threshold_never_reduces_recall(fitted, bundle):
    p = fitted.predict_proba(bundle.val)
    yb = bundle.val["target_state_id"].to_numpy() >= 2
    rec = [(fitted.decide(p, t) >= 2)[yb].mean() for t in (0.05, 0.2, 0.5, 0.8)]
    assert all(a >= b for a, b in zip(rec, rec[1:]))


def test_beta_controls_recall_precision_tradeoff():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.1
    p = np.clip(0.1 + 0.5 * y + rng.normal(0, 0.2, 5000), 0, 1)
    grid = np.linspace(0.02, 0.98, 97)
    t_recall, _ = select_threshold(y, p, 2.0, grid)
    t_precision, _ = select_threshold(y, p, 0.5, grid)
    assert t_recall < t_precision                           # recall-weighted -> lower (more sensitive) threshold


def test_calibration_is_monotone_and_not_worse(fitted, bundle):
    xs = np.linspace(0, 1, 50)
    assert (np.diff(fitted._calibrate_values(xs)) >= -1e-12).all()
    yb = bundle.val["target_state_id"].to_numpy() >= 2
    _, raw = calibration_table(yb, fitted.raw_congestion_score(bundle.val))
    _, cal = calibration_table(yb, fitted.congestion_probability(bundle.val))
    assert cal["brier"] <= raw["brier"] + 1e-9              # in-sample isotonic cannot be worse


def test_save_load_roundtrip(fitted, bundle, small_cfg, tmp_path):
    fitted.save(tmp_path, extra_meta={"x": 1})
    m2 = XGBCongestionModel.load(tmp_path, small_cfg)
    np.testing.assert_allclose(fitted.predict_proba(bundle.val.head(500)), m2.predict_proba(bundle.val.head(500)), atol=1e-6)
    np.testing.assert_allclose(fitted.congestion_probability(bundle.val.head(500)), m2.congestion_probability(bundle.val.head(500)))
    assert m2.threshold == fitted.threshold and m2.feature_columns == fitted.feature_columns
    assert "SIMULATED" in m2.meta["data_note"] and m2.meta["horizon_steps"] == 6


def test_missing_feature_and_unloaded_model(fitted, bundle, small_cfg, tmp_path):
    with pytest.raises(KeyError):
        fitted.predict_proba(bundle.val.drop(columns=[fitted.feature_columns[0]]))
    with pytest.raises(RuntimeError):
        XGBCongestionModel(small_cfg).predict_proba(bundle.val)
    with pytest.raises(FileNotFoundError):
        XGBCongestionModel.load(tmp_path / "nothing", small_cfg)


def test_nan_inputs_are_handled(fitted, bundle):
    d = bundle.val.head(50).copy()
    d.loc[d.index[:5], fitted.feature_columns[3]] = np.nan
    assert np.isfinite(fitted.predict_proba(d)).all()


def test_importance_and_shap(fitted, bundle):
    imp = fitted.importance()
    assert imp["gain_share"].sum() == pytest.approx(1.0)
    assert set(imp["group"]) <= set(bundle.builder.feature_groups)
    shap = fitted.shap_top(bundle.val, k=10, max_rows=300)
    assert len(shap) == 10 and (shap["mean_abs_shap"] >= 0).all()


def test_tuning_returns_valid_params_and_trial_table(small_cfg, bundle):
    m = XGBCongestionModel(small_cfg)
    best, trials = m.tune(bundle.train, bundle.val, bundle.builder.feature_columns)
    assert len(trials) == 2 and trials.loc[0, "trial"] == 0
    assert {"max_depth", "learning_rate", "class_weight_power"} <= set(best)
    assert best["class_weight_power"] in (0.0, 0.5, 1.0) and best["max_depth"] in (4, 6, 8)
    assert trials["best_iteration"].ge(1).all()
    assert trials["val_pr_auc"].between(0, 1).all()
