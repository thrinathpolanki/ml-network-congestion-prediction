import json

import pytest

from src.analysis.eda import indicator_strength, run_eda
from src.data.generator import generate_dataset
from src.features.labeling import add_future_target, label_dataframe


@pytest.fixture(scope="module")
def labelled(cfg):
    df = generate_dataset(cfg, n_sites=6, start="2026-01-05", end="2026-01-26", seed=4,
                          intensity=1.4, inject_issues=False)
    return add_future_target(label_dataframe(df, cfg), cfg)


def test_run_eda_creates_figures_and_summary(cfg, labelled, tmp_path):
    s = run_eda(labelled, cfg, tmp_path)
    assert len(s["figures"]) >= 8
    assert all((tmp_path / f"{p.split('/')[-1]}").exists() for p in s["figures"].values() if p)
    data = json.loads((tmp_path / "eda_summary.json").read_text())
    assert "SIMULATED" in data["data_note"] and data["suggested_peak_hours"]


def test_indicator_strength_ranks_load_kpis_above_noise(labelled):
    t = indicator_strength(labelled, min_events=20)
    top = set(t.head(5)["indicator"])
    assert "bandwidth_utilization_pct" in top or "congestion_score" in top
    assert t["auc_within_site"].between(0, 1).all()


# ------------------------------- updated data: cell positions ---------------------------------------- #
@pytest.fixture(scope="module")
def full_cells(cfg):
    df = generate_dataset(cfg, n_sites=20, start="2026-01-05", end="2026-02-09", seed=42, inject_issues=False)
    return add_future_target(label_dataframe(df, cfg), cfg)


def test_haversine_known_distances():
    from src.analysis.eda import _haversine_km
    assert _haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.19, abs=0.1)
    assert _haversine_km(16.343976, 80.524684, 16.343976, 80.524684) == pytest.approx(0.0, abs=1e-9)


def test_spatial_effect_is_region_not_distance_in_the_simulation(cfg, full_cells):
    """Position is generated FROM region, and cells do not interact: distance must not matter once region is held fixed."""
    from src.analysis.eda import spatial_stats
    sp = spatial_stats(full_cells, cfg)
    assert sp["n_cells"] == 20
    assert sp["rate_vs_distance_spearman_pooled"] < -0.5                       # busy regions sit nearer the centre...
    assert abs(sp["rate_vs_distance_spearman_within_region"]) < 0.4            # ...but within a region distance explains little
    assert abs(sp["corr_vs_distance_spearman_same_region_pairs_region_controlled"]) < 0.4
    assert sp["corr_vs_distance_spearman_same_region_pairs_pooled_confounded"] < -0.5     # the trap: pooled pairs look spatial
    assert sp["mean_hourly_congestion_corr_same_region_pairs"] > sp["mean_hourly_congestion_corr_different_region_pairs"]


def test_run_eda_adds_map_and_degrades_gracefully_without_positions(cfg, full_cells, tmp_path):
    s = run_eda(full_cells, cfg, tmp_path / "with")
    assert "cell_map" in s["figures"] and (tmp_path / "with" / "09a_cell_map.png").exists() and s["spatial"]["n_cells"] == 20
    s2 = run_eda(full_cells.drop(columns=["latitude", "longitude", "location"]), cfg, tmp_path / "without")
    assert "spatial" not in s2 and "cell_map" not in s2["figures"]           # positions are optional metadata


def test_sample_heatmap_shows_every_cell_and_every_step(cfg, tmp_path):
    from src.analysis.eda import fig_sample_heatmap
    from src.data.sample import build_sample
    path = fig_sample_heatmap(build_sample(cfg), tmp_path)
    assert path.endswith("09b_sample_day_heatmap.png") and (tmp_path / "09b_sample_day_heatmap.png").stat().st_size > 20_000
