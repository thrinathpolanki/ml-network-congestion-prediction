import pandas as pd
import pytest

from src.data.generator import generate_dataset
from src.data.schema import REQUIRED_COLUMNS, count_invalid_values, validate_schema


def test_schema_and_provenance(small_df):
    validate_schema(small_df)
    assert set(REQUIRED_COLUMNS) <= set(small_df.columns)
    assert (small_df["data_source"] == "synthetic").all()


def test_reproducible_with_same_seed(cfg):
    kw = dict(n_sites=2, start="2026-01-05", end="2026-01-08", seed=5)
    pd.testing.assert_frame_equal(generate_dataset(cfg, **kw), generate_dataset(cfg, **kw))


def test_different_seed_differs(cfg):
    kw = dict(n_sites=2, start="2026-01-05", end="2026-01-08", inject_issues=False)
    a, b = generate_dataset(cfg, seed=1, **kw), generate_dataset(cfg, seed=2, **kw)
    assert not a["latency_ms"].equals(b["latency_ms"])


def test_clean_data_is_valid_and_regular(small_df):
    assert small_df.isna().sum().sum() == 0
    assert count_invalid_values(small_df) == {}
    assert not small_df.duplicated(["site_id", "timestamp"]).any()
    assert small_df["site_id"].nunique() == 3


def test_configurable_frequency(cfg):
    df = generate_dataset(cfg, n_sites=1, start="2026-01-05", end="2026-01-06",
                          freq="15min", seed=1, inject_issues=False)
    assert len(df) == 96


def test_quality_issues_are_injected(cfg):
    df = generate_dataset(cfg, n_sites=3, start="2026-01-05", end="2026-01-20", seed=1)
    assert df.isna().sum().sum() > 0
    assert df.duplicated(["site_id", "timestamp"]).sum() > 0


def test_intensity_increases_utilisation(cfg):
    kw = dict(n_sites=3, start="2026-01-05", end="2026-01-12", seed=3, inject_issues=False)
    lo, hi = generate_dataset(cfg, intensity=0.6, **kw), generate_dataset(cfg, intensity=1.4, **kw)
    assert hi["bandwidth_utilization_pct"].mean() > lo["bandwidth_utilization_pct"].mean()


def test_congestion_degrades_kpis(small_df):
    hi = small_df["bandwidth_utilization_pct"] > 85
    lo = small_df["bandwidth_utilization_pct"] < 40
    if hi.sum() > 5:
        assert small_df.loc[hi, "latency_ms"].mean() > small_df.loc[lo, "latency_ms"].mean()
        assert small_df.loc[hi, "packet_loss_pct"].mean() > small_df.loc[lo, "packet_loss_pct"].mean()


def test_weekday_evening_peak_exceeds_night(small_df):
    wd = small_df[small_df["timestamp"].dt.dayofweek < 5]
    h = wd.groupby(wd["timestamp"].dt.hour)["bandwidth_utilization_pct"].mean()
    assert h.loc[19:21].max() > 2 * h.loc[2:4].mean() or h.loc[10:11].max() > 2 * h.loc[2:4].mean()


def test_invalid_args(cfg):
    with pytest.raises(ValueError):
        generate_dataset(cfg, n_sites=0)
    with pytest.raises(ValueError):
        generate_dataset(cfg, intensity=-1)
