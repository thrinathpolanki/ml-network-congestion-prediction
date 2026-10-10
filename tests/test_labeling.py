import copy

import numpy as np
import pandas as pd
import pytest

from src.config import horizon_steps
from src.features.labeling import (add_future_target, assign_state_id,
                                   compute_congestion_score, label_dataframe)


def _row(**kw):
    base = dict(active_users=100, download_throughput_mbps=300, bandwidth_utilization_pct=30,
                latency_ms=20, packet_loss_pct=0.05, jitter_ms=2, dropped_call_rate_pct=0.1,
                retransmission_rate_pct=0.5, handover_failure_rate_pct=0.2, cpu_utilization_pct=30)
    base.update(kw)
    return base


def test_healthy_row_is_normal(cfg):
    out = label_dataframe(pd.DataFrame([_row()]), cfg)
    assert out["congestion_state"].iloc[0] == "NORMAL"


def test_all_bad_row_is_critical(cfg):
    bad = _row(download_throughput_mbps=10, bandwidth_utilization_pct=99, latency_ms=400,
               packet_loss_pct=8, jitter_ms=80, dropped_call_rate_pct=9,
               retransmission_rate_pct=20, handover_failure_rate_pct=12, cpu_utilization_pct=99)
    assert label_dataframe(pd.DataFrame([bad]), cfg)["congestion_state"].iloc[0] == "CRITICAL"


def test_score_is_monotonic_in_latency(cfg):
    df = pd.DataFrame([_row(latency_ms=x) for x in (20, 80, 150, 300)])
    s = compute_congestion_score(df, cfg).to_numpy()
    assert (np.diff(s) >= 0).all() and s[-1] > s[0]


def test_thresholds_are_configurable(cfg):
    c2 = copy.deepcopy(cfg)
    row = pd.DataFrame([_row(bandwidth_utilization_pct=80, latency_ms=100, packet_loss_pct=1.5)])
    default_id = int(label_dataframe(row, cfg)["congestion_state_id"].iloc[0])
    c2["labeling"]["thresholds"] = {"warning": 0.01, "congested": 0.02, "critical": 0.03}
    assert int(label_dataframe(row, c2)["congestion_state_id"].iloc[0]) == 3 >= default_id


def test_bad_threshold_order_rejected(cfg):
    c2 = copy.deepcopy(cfg)
    c2["labeling"]["thresholds"] = {"warning": 0.5, "congested": 0.4, "critical": 0.9}
    with pytest.raises(ValueError):
        assign_state_id(pd.Series([0.1]), c2)


def test_nan_indicator_is_ignored_and_all_nan_is_unknown(cfg):
    ok = _row(latency_ms=np.nan)
    assert label_dataframe(pd.DataFrame([ok]), cfg)["congestion_state_id"].iloc[0] == 0
    allnan = {k: np.nan for k in _row()}
    assert label_dataframe(pd.DataFrame([allnan]), cfg)["congestion_state"].iloc[0] == "UNKNOWN"


def test_future_target_is_shifted_per_site(small_df, cfg):
    lab = label_dataframe(small_df, cfg)
    h = horizon_steps(cfg)
    out = add_future_target(lab, cfg)
    site = "CELL_001"
    full = lab[lab.site_id == site].reset_index(drop=True)
    got = out[out.site_id == site].reset_index(drop=True)
    assert len(got) == len(full) - h
    assert (got["target_state_id"].to_numpy() == full["congestion_state_id"].to_numpy()[h:]).all()
    # no leakage across sites: each site loses exactly its last h rows
    assert len(out) == len(lab) - h * lab.site_id.nunique()


def test_horizon_must_divide_sampling(cfg):
    c2 = copy.deepcopy(cfg)
    c2["prediction"]["horizon_minutes"] = 7
    with pytest.raises(ValueError):
        horizon_steps(c2)
