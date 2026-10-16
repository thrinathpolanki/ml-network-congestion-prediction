import copy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.features.engineering import InsufficientHistoryError  # noqa: E402
from src.models.lstm_model import (CongestionNet, LSTMCongestionModel, SequenceBuilder,  # noqa: E402
                                   _class_weights)

L, DYN, STAT = 5, ["a", "b"], ["s"]


def toy(n_sites=2, n=300, seed=0):
    rng, parts = np.random.default_rng(seed), []
    for s in range(n_sites):
        a, b = rng.normal(size=n), rng.normal(size=n)
        d = pd.DataFrame({"site_id": f"S{s}", "timestamp": pd.date_range("2026-01-01", periods=n, freq="5min"),
                          "a": a, "b": b, "s": float(s), "state_now": 0.0})
        d["target_state_id"] = np.digitize(a, [-0.5, 0.5, 1.2])      # learnable from the last step
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def tcfg(cfg):
    c = copy.deepcopy(cfg)
    c["lstm"].update(seq_len=L, hidden_size=8, num_layers=1, batch_size=64, max_epochs=8, patience=8,
                     windows_per_epoch=None, learning_rate=0.01)
    return c


@pytest.fixture(scope="module")
def data():
    f = toy()
    cut = pd.Timestamp("2026-01-01") + pd.Timedelta("5min") * 200
    sb = SequenceBuilder(DYN, STAT, L).fit(f[f.timestamp < cut])
    return f, cut, sb


# ------------------------------ windows ------------------------------------ #
def test_windows_are_exact_slices_and_never_cross_sites(data):
    f, _, sb = data
    ss = sb.build(f)
    z = sb.scaler.transform(f.sort_values(["site_id", "timestamp"]).reset_index(drop=True))
    x, st = ss.arrays(np.arange(len(ss)))
    site = f.sort_values(["site_id", "timestamp"])["site_id"].to_numpy()
    for k in range(0, len(ss), 17):
        e = ss.end_idx[k]
        np.testing.assert_allclose(x[k], z[DYN].to_numpy(np.float32)[e - L + 1:e + 1])
        assert site[e - L + 1] == site[e]                       # same site for the whole window
        assert st[k, 0] == pytest.approx(z["s"].to_numpy(np.float32)[e])


def test_first_rows_of_each_site_are_excluded(data):
    f, _, sb = data
    ss = sb.build(f)
    assert len(ss) == 2 * (300 - (L - 1))
    assert ss.meta.groupby("site_id").size().eq(300 - (L - 1)).all()


def test_windows_with_missing_values_are_dropped(data):
    f, _, sb = data
    g = f.copy()
    g.loc[50, "a"] = np.nan                                    # row 50 of site S0
    ss = sb.build(g)
    assert len(ss) == 2 * (300 - (L - 1)) - L                  # exactly the L windows containing it
    assert not np.isnan(ss.arrays(np.arange(len(ss)))[0]).any()


def test_window_is_causal(data):
    f, _, sb = data
    t0 = 150
    g = f.copy()
    g.loc[g.index[(g.site_id == "S0") & (g.index > t0)], ["a", "b"]] *= 50      # corrupt S0's future
    s1, s2 = sb.build(f), sb.build(g)
    keep = [k for k in range(len(s1)) if s1.meta.site_id[k] == "S0" and s1.end_idx[k] <= t0]
    assert keep
    np.testing.assert_array_equal(s1.arrays(np.array(keep))[0], s2.arrays(np.array(keep))[0])


def test_scaler_uses_training_period_only(data):
    f, cut, sb = data
    assert sb.scaler.scaler.mean_[0] == pytest.approx(f.loc[f.timestamp < cut, "a"].mean())
    assert sb.scaler.scaler.mean_[0] != pytest.approx(f["a"].mean())


def test_rows_argument_selects_samples_and_targets(data):
    f, cut, sb = data
    val = f[f.timestamp >= cut]
    ss = sb.build(f, val)
    assert len(ss) == len(val) and (ss.y == val.sort_values(["site_id", "timestamp"])["target_state_id"].to_numpy()).all()


def test_inference_window_matches_training_window(data):
    f, _, sb = data
    ss = sb.build(f)
    k = 120                                                    # some sample of site S0
    site, e = ss.meta.site_id[k], ss.end_idx[k]
    hist = f[(f.site_id == site)].sort_values("timestamp").iloc[: e + 1]
    dyn, stat = sb.window_from_features(hist)
    x, st = ss.arrays(np.array([k]))
    np.testing.assert_allclose(dyn, x, atol=1e-6)
    np.testing.assert_allclose(stat, st, atol=1e-6)


def test_inference_window_errors(data):
    f, _, sb = data
    one = f[f.site_id == "S0"]
    with pytest.raises(InsufficientHistoryError):
        sb.window_from_features(one.head(L - 1))
    bad = one.head(20).copy(); bad.iloc[-1, bad.columns.get_loc("a")] = np.nan
    with pytest.raises(InsufficientHistoryError):
        sb.window_from_features(bad)
    with pytest.raises(ValueError):
        sb.window_from_features(f.head(400))                   # two sites


# ------------------------------ network / training ------------------------- #
def test_network_shapes():
    out = CongestionNet(2, 1, hidden=8)(torch.zeros(7, L, 2), torch.zeros(7, 1))
    assert out.shape == (7, 4)


def test_class_weights_balance_at_power_one():
    y = np.array([0] * 90 + [1] * 5 + [2] * 4 + [3])
    w = _class_weights(y, 1.0).numpy()
    counts = np.bincount(y)
    assert np.allclose(w * counts, (w * counts)[0])


@pytest.fixture(scope="module")
def trained(tcfg, data, tmp_path_factory):
    f, cut, sb = data
    tr, va = sb.build(f, f[f.timestamp < cut]), sb.build(f, f[f.timestamp >= cut])
    m = LSTMCongestionModel(tcfg, DYN, STAT)
    ck = tmp_path_factory.mktemp("ck") / "best.pt"
    hist = m.fit(tr, va, checkpoint_path=ck)
    m.calibrate(va); m.fit_threshold(va)
    return m, tr, va, hist, ck


def test_training_learns_and_checkpoints(trained):
    m, tr, va, hist, ck = trained
    assert hist.val_loss.iloc[-1] < hist.val_loss.iloc[0] and ck.exists()
    assert (m.predict(va, use_threshold=False) == va.y).mean() > 0.8
    assert m.meta["best_epoch"] == int(hist.loc[hist.val_loss.idxmin(), "epoch"])
    p = m.predict_proba(va)
    assert p.shape == (len(va), 4) and np.allclose(p.sum(1), 1, atol=1e-5)


def test_training_is_reproducible(tcfg, data):
    f, cut, sb = data
    tr, va = sb.build(f, f[f.timestamp < cut]), sb.build(f, f[f.timestamp >= cut])
    a, b = LSTMCongestionModel(tcfg, DYN, STAT), LSTMCongestionModel(tcfg, DYN, STAT)
    a.fit(tr, va); b.fit(tr, va)
    np.testing.assert_allclose(a.predict_proba(va), b.predict_proba(va), atol=1e-5)


def test_lower_threshold_never_reduces_recall(trained):
    m, _, va, _, _ = trained
    p = m.predict_proba(va)
    yb = va.y >= 2
    rec = [(m.decide(p, t) >= 2)[yb].mean() for t in (0.05, 0.3, 0.6, 0.9)]
    assert all(a >= b for a, b in zip(rec, rec[1:]))


def test_save_load_roundtrip(trained, tcfg, data, tmp_path):
    m, _, va, _, _ = trained
    _, _, sb = data
    m.save(tmp_path, sb)
    m2, sb2 = LSTMCongestionModel.load(tmp_path, tcfg)
    np.testing.assert_allclose(m.predict_proba(va), m2.predict_proba(va), atol=1e-6)
    assert m2.threshold == m.threshold and sb2.seq_len == L
    np.testing.assert_allclose(sb.scaler.scaler.mean_, sb2.scaler.scaler.mean_)
    assert "SIMULATED" in m2.meta["data_note"]


def test_load_missing_artifacts(tcfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        LSTMCongestionModel.load(tmp_path / "nope", tcfg)
