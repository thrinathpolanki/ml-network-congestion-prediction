import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.data.generator import generate_dataset
from src.data.schema import RAW_METRICS, SchemaError, count_invalid_values
from src.features.labeling import add_future_target, label_dataframe
from src.preprocessing.cleaning import clean_dataframe, sanitize_metrics
from src.preprocessing.scaling import FeatureTransformer
from src.preprocessing.splitting import chronological_split


@pytest.fixture(scope="module")
def dirty(cfg):
    return generate_dataset(cfg, n_sites=3, start="2026-01-05", end="2026-01-20", seed=3)


@pytest.fixture(scope="module")
def cleaned(cfg, dirty):
    return clean_dataframe(dirty, cfg)


# ------------------------------- cleaning ---------------------------------- #
def test_no_duplicates_and_regular_grid(cleaned):
    df, _ = cleaned
    assert not df.duplicated(["site_id", "timestamp"]).any()
    for _, g in df.groupby("site_id"):
        assert g["timestamp"].is_monotonic_increasing
        assert (g["timestamp"].diff().dropna() == pd.Timedelta("5min")).all()


def test_invalid_values_removed_and_reported(cleaned, dirty):
    df, rep = cleaned
    assert count_invalid_values(df) == {}
    assert count_invalid_values(dirty)  # the dirty input really had some
    assert rep["duplicates_removed"] > 0 and rep["cells_imputed"] > 0


def test_short_gaps_filled_long_gaps_kept(cfg):
    base = generate_dataset(cfg, n_sites=1, start="2026-01-05", end="2026-01-07", seed=1, inject_issues=False)
    df = base.copy()
    df.loc[10:12, "latency_ms"] = np.nan          # 3-step gap -> filled
    df.loc[50:59, "latency_ms"] = np.nan          # 10-step gap (> 6) -> stays NaN
    out, _ = clean_dataframe(df, cfg)
    assert out.loc[10:12, "latency_ms"].notna().all()
    assert out.loc[50:59, "latency_ms"].isna().all()
    assert out.loc[10:12, "is_imputed"].all()


def test_isolated_glitch_removed_but_genuine_episode_kept(cfg):
    df = generate_dataset(cfg, n_sites=1, start="2026-01-05", end="2026-01-07", seed=2, inject_issues=False)
    df["latency_ms"] = 30.0
    df.loc[100, "latency_ms"] = 600.0              # one-sample glitch
    df.loc[200:204, "latency_ms"] = 150.0          # real 25-minute congestion episode
    out, rep = clean_dataframe(df, cfg)
    assert out.loc[100, "latency_ms"] < 100
    assert (out.loc[200:204, "latency_ms"] == 150.0).all()
    assert rep["outliers_removed"].get("latency_ms", 0) >= 1


def test_missing_rows_are_reinserted(cfg):
    df = generate_dataset(cfg, n_sites=1, start="2026-01-05", end="2026-01-06", seed=1, inject_issues=False)
    out, rep = clean_dataframe(df.drop(index=[20, 21]), cfg)
    assert len(out) == len(df) and rep["rows_inserted_for_gaps"] == 2


def test_schema_error_on_missing_column(cfg, small_df):
    with pytest.raises(SchemaError):
        clean_dataframe(small_df.drop(columns=["latency_ms"]), cfg)


def test_sanitize_single_reading():
    row = pd.DataFrame([{"latency_ms": "abc", "bandwidth_utilization_pct": 150, "packet_loss_pct": 0.2}])
    out, n = sanitize_metrics(row)
    assert out["latency_ms"].isna().all() and out["bandwidth_utilization_pct"].isna().all()
    assert out["packet_loss_pct"].iloc[0] == 0.2 and "bandwidth_utilization_pct" in n


def test_clean_is_idempotent(cfg, cleaned):
    df, _ = cleaned
    again, rep = clean_dataframe(df.drop(columns=["is_imputed"]), cfg)
    assert rep["duplicates_removed"] == 0 and rep["rows_inserted_for_gaps"] == 0


# ------------------------------- splitting --------------------------------- #
@pytest.fixture(scope="module")
def labelled(cfg, cleaned):
    return add_future_target(label_dataframe(cleaned[0], cfg), cfg)


def test_split_is_chronological_and_disjoint(cfg, labelled):
    s = chronological_split(labelled, cfg)
    assert s.train["timestamp"].max() < s.val["timestamp"].min()
    assert s.val["timestamp"].max() < s.test["timestamp"].min()
    assert len(s.train) + len(s.val) + len(s.test) + sum(s.purged_rows.values()) == len(labelled)


def test_purge_prevents_target_leakage(cfg, labelled):
    s = chronological_split(labelled, cfg)
    h = pd.Timedelta("5min") * s.boundaries["purge_steps"]
    t1, t2 = pd.Timestamp(s.boundaries["train_end_exclusive"]), pd.Timestamp(s.boundaries["val_end_exclusive"])
    assert (s.train["timestamp"] + h < t1).all()      # train targets lie inside train period
    assert (s.val["timestamp"] + h < t2).all()        # val targets lie inside val period


def test_split_fractions_roughly_respected(cfg, labelled):
    s = chronological_split(labelled, cfg)
    n = len(labelled)
    assert 0.65 < len(s.train) / n < 0.72 and 0.12 < len(s.val) / n < 0.17


def test_bad_fractions_rejected(cfg, labelled):
    c = copy.deepcopy(cfg)
    c["preprocessing"]["split"] = {"train_frac": 0.9, "val_frac": 0.2}
    with pytest.raises(ValueError):
        chronological_split(labelled, c)


# ------------------------------- scaling ----------------------------------- #
def test_transformer_fits_on_train_only(cleaned):
    df = cleaned[0]
    cols = ["latency_ms", "packet_loss_pct"]
    tr, te = df.iloc[:500], df.iloc[500:]
    ft = FeatureTransformer(cols, ["region"]).fit(tr)
    z = ft.transform(tr)
    assert np.allclose(z["latency_ms"].mean(), 0, atol=1e-6)
    assert ft.scaler.mean_[0] == pytest.approx(tr["latency_ms"].mean())   # not influenced by test
    assert ft.transform(te).shape[1] == len(ft.feature_names_out)


def test_transformer_roundtrip_and_unknown_category(cleaned, tmp_path):
    df = cleaned[0]
    ft = FeatureTransformer(["latency_ms"], ["region"]).fit(df)
    p = ft.save(tmp_path / "t.joblib")
    ft2 = FeatureTransformer.load(p)
    pd.testing.assert_frame_equal(ft.transform(df.head(20)), ft2.transform(df.head(20)))
    odd = df.head(3).copy()
    odd["region"] = "mars"
    assert (ft2.transform(odd).filter(like="region__").to_numpy() == 0).all()


def test_transformer_errors(cleaned):
    ft = FeatureTransformer(["latency_ms"])
    with pytest.raises(NotFittedError):
        ft.transform(cleaned[0])
    ft.fit(cleaned[0])
    with pytest.raises(KeyError):
        ft.transform(cleaned[0].drop(columns=["latency_ms"]))


# ------------------------------- updated data: positions + the clean sample --------------------------- #
def test_location_columns_survive_cleaning_including_reinserted_gap_rows(cfg, dirty, cleaned):
    out, rep = cleaned
    assert rep["rows_inserted_for_gaps"] > 0                                   # gap rows were rebuilt...
    assert out[["latitude", "longitude", "location"]].notna().all().all()      # ...and still know where they are
    original = dirty.groupby("site_id")["location"].first()
    assert (out.groupby("site_id")["location"].nunique() == 1).all()
    assert out.groupby("site_id")["location"].first().to_dict() == original.to_dict()


def test_cleaning_a_clean_sample_is_virtually_a_noop(cfg):
    """A dataset with nothing wrong with it must come out (almost) exactly as it went in."""
    from src.data.sample import build_sample
    s = build_sample(cfg).sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    out, rep = clean_dataframe(s, cfg)
    out = out.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    assert rep["rows_in"] == rep["rows_out"] == len(s) and rep["duplicates_removed"] == 0
    assert rep["rows_inserted_for_gaps"] == 0 and rep["invalid_values_nullified"] == {}
    assert (out[["site_id", "timestamp"]].to_numpy() == s[["site_id", "timestamp"]].to_numpy()).all()
    assert (out[["latitude", "longitude", "location"]] == s[["latitude", "longitude", "location"]]).all().all()
    changed = sum(int((~np.isclose(s[c], out[c], rtol=1e-9)).sum()) for c in RAW_METRICS)
    assert changed <= 0.0001 * len(s) * len(RAW_METRICS)                       # false-positive rate of the glitch filter < 0.01 %


def test_clean_data_has_no_impossible_values_before_cleaning(cfg):
    """Regression: the generator once emitted negative jitter, which cleaning then (wrongly) had to repair."""
    df = generate_dataset(cfg, n_sites=20, start="2026-01-01", end="2026-02-15", inject_issues=False)
    assert count_invalid_values(df) == {}
    _, rep = clean_dataframe(df, cfg)
    assert rep["invalid_values_nullified"] == {}


def test_prepare_pipeline_writes_where_told_and_never_clobbers_the_default(cfg, tmp_path):
    from src.data.generator import save_dataset
    from src.data.sample import build_sample
    from src.pipeline.prepare import run_prepare
    c = copy.deepcopy(cfg)
    c["paths"].update(processed_dir=str(tmp_path / "processed"), reports_dir=str(tmp_path / "reports"))
    src_csv = save_dataset(build_sample(c), tmp_path / "sample.csv", c)
    out = tmp_path / "elsewhere" / "clean_sample.csv"
    r = run_prepare(c, src_csv, out)
    assert out.exists() and not (tmp_path / "processed" / c["preprocessing"]["clean_filename"]).exists()
    assert r["report_file"].endswith("preprocessing_report__clean_sample.json") and (tmp_path / "reports" / "preprocessing_report__clean_sample.json").exists()
    assert r["location"]["rows_missing_position"] == 0 and r["location"]["cells_with_one_fixed_position"] is True
    assert r["cleaning"]["rows_in"] == r["cleaning"]["rows_out"]
    s = r["split"]
    assert s["train"]["end"] < s["val"]["start"] < s["val"]["end"] < s["test"]["start"]       # still chronological on the 2-day sample
