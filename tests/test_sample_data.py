import copy

import numpy as np
import pandas as pd
import pytest

from src.data.generator import generate_dataset, site_locations
from src.data.sample import LOCATION_RE, build_sample, validate_sample
from src.data.schema import RAW_METRICS, count_invalid_values


@pytest.fixture(scope="module")
def sample(cfg):
    return build_sample(cfg)


def test_sample_passes_every_self_check(cfg, sample):
    failed = {k: d for k, (ok, d) in validate_sample(sample, cfg).items() if not ok}
    assert failed == {}


def test_sample_is_20_cells_on_a_perfect_5_minute_grid(cfg, sample):
    assert sample["site_id"].nunique() == 20
    per_cell = sample.groupby("site_id").size()
    assert (per_cell == 2 * 288).all()                                    # 2 full days x 288 intervals
    for _, g in sample.groupby("site_id"):
        assert (g.sort_values("timestamp")["timestamp"].diff().dropna() == pd.Timedelta("5min")).all()
        assert g["timestamp"].min() == pd.Timestamp("2026-02-24 00:00") and g["timestamp"].max() == pd.Timestamp("2026-02-25 23:55")
    assert sample.isna().sum().sum() == 0 and not sample.duplicated(["site_id", "timestamp"]).any()
    assert count_invalid_values(sample) == {}


def test_location_columns_in_the_requested_format(sample):
    assert {"latitude", "longitude", "location"} <= set(sample.columns)
    assert sample["location"].map(lambda v: bool(LOCATION_RE.match(v))).all()
    row = sample.iloc[0]
    assert row["location"] == f"{row['latitude']:.6f},{row['longitude']:.6f}"
    first = sample.groupby("site_id")[["latitude", "longitude"]].first()
    assert first["latitude"].between(15.9, 16.8).all() and first["longitude"].between(80.1, 80.9).all()   # around the centre


def test_each_cell_has_one_fixed_distinct_position(sample):
    assert (sample.groupby("site_id")["location"].nunique() == 1).all()
    assert sample.groupby("site_id")["location"].first().nunique() == 20


def test_denser_regions_sit_closer_to_the_centre(cfg):
    df = generate_dataset(cfg, n_sites=60, start="2026-01-05", end="2026-01-06", inject_issues=False)
    c = cfg["generator"]["location"]
    f = df.groupby("site_id")[["region", "latitude", "longitude"]].first()
    km = np.hypot((f.latitude - c["center_lat"]) * 111.32, (f.longitude - c["center_lon"]) * 111.32 * np.cos(np.radians(c["center_lat"])))
    by = km.groupby(f.region).mean()
    assert by["urban_core"] < by["suburban"] < by["rural"]
    assert all(km[f.region == r].max() <= c["radius_km"][r] * 1.01 for r in c["radius_km"] if (f.region == r).any())


def test_coordinates_do_not_change_any_simulated_kpi(cfg):
    """Regression guard: positions use an independent seed, so moving the map centre must not alter a single KPI value."""
    kw = dict(n_sites=5, start="2026-01-05", end="2026-01-08", seed=11, inject_issues=True)
    moved = copy.deepcopy(cfg)
    moved["generator"]["location"].update(center_lat=28.6139, center_lon=77.2090)
    a, b = generate_dataset(cfg, **kw), generate_dataset(moved, **kw)
    pd.testing.assert_frame_equal(a[["timestamp", "site_id", "region", "site_capacity_mbps"] + RAW_METRICS],
                                  b[["timestamp", "site_id", "region", "site_capacity_mbps"] + RAW_METRICS])
    assert not a["latitude"].equals(b["latitude"])                          # but the positions did move


def test_positions_are_reproducible_and_seed_dependent(cfg):
    regions = ["urban_core", "rural", "suburban"]
    a = site_locations(cfg["generator"]["location"], regions, 42)
    assert a == site_locations(cfg["generator"]["location"], regions, 42)
    assert a != site_locations(cfg["generator"]["location"], regions, 43)


def test_sample_is_reproducible(cfg, sample):
    pd.testing.assert_frame_equal(sample, build_sample(cfg))


def test_main_dataset_may_still_be_dirty_but_keeps_location(cfg):
    df = generate_dataset(cfg, n_sites=3, start="2026-01-05", end="2026-01-20", seed=1)           # issues injected
    assert df.duplicated(["site_id", "timestamp"]).sum() > 0 and df["location"].notna().all()
