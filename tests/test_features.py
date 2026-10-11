import copy
import logging

import numpy as np
import pandas as pd
import pytest

from src.data.generator import generate_dataset
from src.data.schema import RAW_METRICS, SchemaError
from src.features.dataset import build_dataset
from src.features.engineering import KEYS, FeatureBuilder, InsufficientHistoryError
from src.features.labeling import add_future_target, label_dataframe


@pytest.fixture(scope="module")
def df(cfg):
    return generate_dataset(cfg, n_sites=3, start="2026-01-05", end="2026-01-12",
                            seed=9, intensity=1.3, inject_issues=False)


@pytest.fixture(scope="module")
def builder(cfg, df):
    return FeatureBuilder(cfg).fit(df[df.timestamp < "2026-01-09"])


def _aligned(a: pd.DataFrame, b: pd.DataFrame, cols):
    m = a.merge(b, on=KEYS, suffixes=("_a", "_b"))
    return m, [m[c + "_a"].to_numpy(float) for c in cols], [m[c + "_b"].to_numpy(float) for c in cols]


# --------------------------- registry / hygiene ---------------------------- #
def test_columns_unique_and_present(builder, df):
    cols = builder.feature_columns
    assert len(cols) == len(set(cols)) > 100
    out = builder.transform(df)
    assert set(cols) <= set(out.columns)
    assert not out[cols].isna().any(axis=None)
    assert all(str(t) == "float32" for t in out[cols].dtypes)


def test_no_target_or_identifier_in_features(builder):
    banned = {"timestamp", "site_id", "region", "is_imputed", "target_state_id",
              "target_score", "target_binary", "congestion_state_id"}
    assert not banned & set(builder.feature_columns)
    assert not any(c.startswith("target_") for c in builder.feature_columns)


def test_describe_lists_every_group(builder):
    md = builder.describe()
    assert all(g in md for g in builder.feature_groups)
    assert "Never used as inputs" in md


# ------------------------------ LEAKAGE PROOFS ----------------------------- #
CUT = pd.Timestamp("2026-01-09 12:00")


def test_causality_removing_the_future_changes_nothing(builder, df):
    full = builder.transform(df, drop_incomplete=False)
    trunc = builder.transform(df[df.timestamp <= CUT], drop_incomplete=False)
    m, a, b = _aligned(full, trunc, builder.feature_columns)
    assert len(m) == len(trunc)
    for fa, fb, name in zip(a, b, builder.feature_columns):
        assert np.allclose(fa, fb, equal_nan=True, rtol=1e-6, atol=1e-6), f"feature {name} depends on the future"


def test_causality_corrupting_the_future_changes_nothing(builder, df):
    bad = df.copy()
    fut = bad.timestamp > CUT
    rng = np.random.default_rng(0)
    for c in RAW_METRICS:
        bad.loc[fut, c] = bad.loc[fut, c] * rng.uniform(3, 9, fut.sum())
    full = builder.transform(df, drop_incomplete=False)
    corrupted = builder.transform(bad, drop_incomplete=False)
    past_a, past_b = full[full.timestamp <= CUT], corrupted[corrupted.timestamp <= CUT]
    m, a, b = _aligned(past_a, past_b, builder.feature_columns)
    for fa, fb, name in zip(a, b, builder.feature_columns):
        assert np.allclose(fa, fb, equal_nan=True, rtol=1e-6, atol=1e-6), f"feature {name} leaks future"


def test_features_are_not_the_target(builder, cfg, df):
    """No feature may equal the label state at t+H (cheap sanity guard)."""
    out = builder.transform(df)
    lab = add_future_target(label_dataframe(df, cfg), cfg)[KEYS + ["target_state_id"]]
    m = out.merge(lab, on=KEYS)
    for c in builder.feature_columns:
        assert not np.array_equal(m[c].to_numpy(), m["target_state_id"].to_numpy()), c


def test_window_parity_inference_matches_training(builder, df):
    full = builder.transform(df, drop_incomplete=False)
    site = "CELL_002"
    g = df[df.site_id == site].sort_values("timestamp").reset_index(drop=True)
    for i in (150, 400, 900, 1500):
        hist = g.iloc[: i + 1]
        live = builder.transform_latest(hist)
        ref = full[(full.site_id == site) & (full.timestamp == g.timestamp[i])]
        np.testing.assert_allclose(live[builder.feature_columns].to_numpy(float),
                                   ref[builder.feature_columns].to_numpy(float),
                                   rtol=2e-3, atol=2e-3)


def test_insufficient_history_rejected(builder, df):
    g = df[df.site_id == "CELL_001"].sort_values("timestamp").head(builder.required_history_steps - 1)
    with pytest.raises(InsufficientHistoryError):
        builder.transform_latest(g)


def test_transform_latest_requires_single_site(builder, df):
    with pytest.raises(ValueError):
        builder.transform_latest(df.groupby("site_id").head(200))


# ------------------------------- semantics --------------------------------- #
def test_time_features_and_forecast_time(builder, df):
    out = builder.transform(df)
    r = out[(out.timestamp == "2026-01-07 19:45") & (out.site_id == "CELL_001")].iloc[0]   # Wednesday
    assert r["hour"] == 19 and r["is_weekend"] == 0 and r["is_peak_hour"] == 1
    # forecast time = 20:15
    assert r["fc_tod_sin"] == pytest.approx(np.sin(2 * np.pi * 20.25 / 24), abs=1e-5)
    sat = out[(out.timestamp == "2026-01-10 10:00") & (out.site_id == "CELL_001")].iloc[0]
    assert sat["is_weekend"] == 1


def test_forecast_features_follow_horizon(cfg, df):
    c2 = copy.deepcopy(cfg)
    c2["prediction"]["horizon_minutes"] = 60
    b2 = FeatureBuilder(c2).fit(df[df.timestamp < "2026-01-09"])
    a = FeatureBuilder(cfg).fit(df[df.timestamp < "2026-01-09"]).transform(df)
    b = b2.transform(df)
    m = a.merge(b, on=KEYS, suffixes=("_a", "_b"))
    assert not np.allclose(m["fc_tod_sin_a"], m["fc_tod_sin_b"])
    assert np.allclose(m["tod_sin_a"], m["tod_sin_b"])


def test_state_now_matches_labeling(builder, cfg, df):
    out = builder.transform(df)
    lab = label_dataframe(df, cfg)[KEYS + ["congestion_state_id"]]
    m = out.merge(lab, on=KEYS, suffixes=("", "_l"))
    assert (m["state_now"].to_numpy() == m["congestion_state_id_l"].to_numpy()).all()


def test_site_relative_removes_region_latency_offset(builder, df):
    out = builder.transform(df)
    med = out.groupby("site_id")["latency_vs_site"].median()
    assert med.between(0.5, 1.5).all()


# ----------------------------- fitted state -------------------------------- #
def test_baselines_come_from_training_rows_only(cfg, df):
    train = df[df.timestamp < "2026-01-09"]
    b = FeatureBuilder(cfg).fit(train)
    expected = train.groupby("site_id")["latency_ms"].median()
    pd.testing.assert_series_equal(b.baselines.table["site_med_latency_ms"], expected,
                                   check_names=False)
    assert b.baselines.table["site_med_latency_ms"].ne(
        df.groupby("site_id")["latency_ms"].median()).any()   # differs from all-data median


def test_unknown_site_uses_fallback(builder, df, caplog):
    d = df[df.site_id == "CELL_001"].copy()
    d["site_id"] = "CELL_999"
    with caplog.at_level(logging.WARNING):
        out = builder.transform(d)
    assert len(out) > 0 and not out[builder.feature_columns].isna().any(axis=None)
    assert "Unknown sites" in caplog.text


def test_irregular_grid_and_missing_columns_rejected(builder, df):
    with pytest.raises(ValueError):
        builder.transform(df.drop(index=df.index[100]))
    with pytest.raises(SchemaError):
        builder.transform(df.drop(columns=["latency_ms"]))
    with pytest.raises(RuntimeError):
        FeatureBuilder(copy.deepcopy(builder.cfg)).transform(df)   # not fitted


def test_save_load_roundtrip(builder, df, tmp_path):
    p = builder.save(tmp_path / "fb.joblib")
    b2 = FeatureBuilder.load(p)
    pd.testing.assert_frame_equal(builder.transform(df), b2.transform(df))
    assert b2.required_history_steps == builder.required_history_steps


# ------------------------------- dataset ----------------------------------- #
@pytest.fixture(scope="module")
def bundle(cfg):
    d = generate_dataset(cfg, n_sites=4, start="2026-01-05", end="2026-01-26", seed=5,
                         intensity=1.3, inject_issues=False)
    return d, build_dataset(d, cfg)


def test_dataset_splits_are_chronological_and_complete(bundle):
    _, b = bundle
    assert b.train.timestamp.max() < b.val.timestamp.min() < b.val.timestamp.max() < b.test.timestamp.min()
    for part in (b.train, b.val, b.test):
        assert not part[b.builder.feature_columns].isna().any(axis=None)
        assert part["target_state_id"].between(0, 3).all()
    assert b.report["n_features"] == len(b.builder.feature_columns)


def test_pipeline_not_influenced_by_validation_or_test_data(cfg, bundle):
    d, b = bundle
    t1 = pd.Timestamp(b.report["baselines_fitted_on_rows_before"])
    changed = d.copy()
    later = changed.timestamp >= t1
    changed.loc[later, "latency_ms"] *= 3
    changed.loc[later, "active_users"] *= 2
    b2 = build_dataset(changed, cfg)
    pd.testing.assert_frame_equal(b.builder.baselines.table, b2.builder.baselines.table)


def test_target_is_state_h_steps_ahead(cfg, bundle):
    d, b = bundle
    lab = label_dataframe(d, cfg)
    site = b.train.site_id.iloc[0]
    g = lab[lab.site_id == site].set_index("timestamp")["congestion_state_id"]
    row = b.train[b.train.site_id == site].iloc[50]
    t_future = row.timestamp + pd.Timedelta(minutes=cfg["prediction"]["horizon_minutes"])
    assert row.target_state_id == g.loc[t_future]


def test_baselines_remember_each_sites_region(builder, df):
    sr = builder.baselines.site_region
    assert set(sr) == set(builder.baselines.table.index)
    assert sr == df.drop_duplicates("site_id").set_index("site_id")["region"].astype(str).to_dict()


# ------------------------------- updated data: cell positions are metadata, never inputs ---------------- #
POSITION_COLS = ["latitude", "longitude", "location"]


def test_positions_are_never_model_inputs(builder):
    cols = builder.feature_columns
    assert not set(POSITION_COLS) & set(cols)
    assert not any(c.startswith(("latitude", "longitude", "location", "dist_km", "nbr_")) for c in cols)   # (not "lat": latency_ms is legitimate)
    assert any(c.startswith("latency_ms") for c in cols)                                                   # guards the guard


def test_features_do_not_depend_on_coordinates(builder, df):
    """Scrambling every position must not change a single feature value."""
    assert set(POSITION_COLS) <= set(df.columns)
    scrambled = df.copy()
    rng = np.random.default_rng(0)
    scrambled["latitude"] = rng.uniform(-80, 80, len(df))
    scrambled["longitude"] = rng.uniform(-170, 170, len(df))
    scrambled["location"] = scrambled["latitude"].map("{:.6f}".format) + "," + scrambled["longitude"].map("{:.6f}".format)
    pd.testing.assert_frame_equal(builder.transform(df), builder.transform(scrambled))


def test_builder_works_when_the_data_source_has_no_positions(builder, df):
    """Real telecom exports may have no coordinates: the pipeline must not require them."""
    bare = df.drop(columns=POSITION_COLS)
    pd.testing.assert_frame_equal(builder.transform(df), builder.transform(bare))
    assert FeatureBuilder(builder.cfg).fit(bare[bare.timestamp < "2026-01-09"]).feature_columns == builder.feature_columns
