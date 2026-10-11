# Feature dictionary

Horizon: 6 steps (0 days 00:30:00). Sampling step: 0 days 00:05:00. Required history at inference: 73 steps.

| Group | # | Description | Uses data from | Leakage control |
|---|---|---|---|---|
| `current_kpis` | 15 | Raw KPIs reported at time t (+ per-user throughput). | t | None: measured at t. |
| `lags` | 25 | Key KPIs [1, 2, 3, 6, 12] steps ago. | t-k | Past only. |
| `rate_of_change` | 15 | x[t] - x[t-k], k in [1, 3, 6]. | t, t-k | Past only. |
| `rolling` | 60 | mean/std/max/min over trailing windows [6, 12, 36] (incl. t). | <= t | Trailing windows (never centred); strict min_periods. |
| `ewm` | 10 | Exponentially weighted mean, spans [3, 12]. | <= t | Recursive filter: causal. |
| `congestion_history` | 10 | Current/previous congestion state & score, share of recent time congested. | <= t | Built from KPIs at <= t via the labeling function; NEVER the target t+H. |
| `calendar_now` | 7 | Hour, time-of-day and day-of-week encodings, weekend, peak-hour flag. | t | Calendar: known. |
| `calendar_forecast_time` | 4 | Same calendar flags for the forecast time t+H. | t (calendar of t+H) | Calendar is known in advance; carries no measurement from the future. |
| `site_static` | 4 | Capacity + per-site typical levels (median latency, p90 utilisation, p90 users). | fitted | Fitted on TRAINING period only; contain no labels. |
| `site_relative` | 3 | Current KPI divided by the site's own baseline (fixes the region latency confound). | t + fitted | As above. |
| `region_onehot` | 5 | One-hot region (categories from training). | t | None. |

**Total features: 158.**

**Never used as inputs:** `target_state_id`, `target_score`, `target_binary`, `timestamp`, `site_id`, `is_imputed` (metadata only), and the cell position columns `latitude`, `longitude`, `location` (descriptive metadata for maps; a validation experiment found no benefit from adding them, see docs/SPATIAL_FEATURES.md).

### current_kpis
`active_users`, `active_sessions`, `traffic_volume_gb`, `bandwidth_utilization_pct`, `download_throughput_mbps`, `upload_throughput_mbps`, `latency_ms`, `packet_loss_pct`, `jitter_ms`, `signal_quality_sinr_db`, `cpu_utilization_pct`, `dropped_call_rate_pct`, `handover_failure_rate_pct`, `retransmission_rate_pct`, `dl_throughput_per_user_mbps`

### lags
`bandwidth_utilization_pct_lag1`, `bandwidth_utilization_pct_lag2`, `bandwidth_utilization_pct_lag3`, `bandwidth_utilization_pct_lag6`, `bandwidth_utilization_pct_lag12`, `latency_ms_lag1`, `latency_ms_lag2`, `latency_ms_lag3`, `latency_ms_lag6`, `latency_ms_lag12`, `packet_loss_pct_lag1`, `packet_loss_pct_lag2`, `packet_loss_pct_lag3`, `packet_loss_pct_lag6`, `packet_loss_pct_lag12`, `dl_throughput_per_user_mbps_lag1`, `dl_throughput_per_user_mbps_lag2`, `dl_throughput_per_user_mbps_lag3`, `dl_throughput_per_user_mbps_lag6`, `dl_throughput_per_user_mbps_lag12`, `active_users_lag1`, `active_users_lag2`, `active_users_lag3`, `active_users_lag6`, `active_users_lag12`

### rate_of_change
`bandwidth_utilization_pct_diff1`, `bandwidth_utilization_pct_diff3`, `bandwidth_utilization_pct_diff6`, `latency_ms_diff1`, `latency_ms_diff3`, `latency_ms_diff6`, `packet_loss_pct_diff1`, `packet_loss_pct_diff3`, `packet_loss_pct_diff6`, `dl_throughput_per_user_mbps_diff1`, `dl_throughput_per_user_mbps_diff3`, `dl_throughput_per_user_mbps_diff6`, `active_users_diff1`, `active_users_diff3`, `active_users_diff6`

### rolling
`bandwidth_utilization_pct_roll6_mean`, `bandwidth_utilization_pct_roll6_std`, `bandwidth_utilization_pct_roll6_max`, `bandwidth_utilization_pct_roll6_min`, `bandwidth_utilization_pct_roll12_mean`, `bandwidth_utilization_pct_roll12_std`, `bandwidth_utilization_pct_roll12_max`, `bandwidth_utilization_pct_roll12_min`, `bandwidth_utilization_pct_roll36_mean`, `bandwidth_utilization_pct_roll36_std`, `bandwidth_utilization_pct_roll36_max`, `bandwidth_utilization_pct_roll36_min`, `latency_ms_roll6_mean`, `latency_ms_roll6_std`, `latency_ms_roll6_max`, `latency_ms_roll6_min`, `latency_ms_roll12_mean`, `latency_ms_roll12_std`, `latency_ms_roll12_max`, `latency_ms_roll12_min`, `latency_ms_roll36_mean`, `latency_ms_roll36_std`, `latency_ms_roll36_max`, `latency_ms_roll36_min`, `packet_loss_pct_roll6_mean`, `packet_loss_pct_roll6_std`, `packet_loss_pct_roll6_max`, `packet_loss_pct_roll6_min`, `packet_loss_pct_roll12_mean`, `packet_loss_pct_roll12_std`, `packet_loss_pct_roll12_max`, `packet_loss_pct_roll12_min`, `packet_loss_pct_roll36_mean`, `packet_loss_pct_roll36_std`, `packet_loss_pct_roll36_max`, `packet_loss_pct_roll36_min`, `dl_throughput_per_user_mbps_roll6_mean`, `dl_throughput_per_user_mbps_roll6_std`, `dl_throughput_per_user_mbps_roll6_max`, `dl_throughput_per_user_mbps_roll6_min`, `dl_throughput_per_user_mbps_roll12_mean`, `dl_throughput_per_user_mbps_roll12_std`, `dl_throughput_per_user_mbps_roll12_max`, `dl_throughput_per_user_mbps_roll12_min`, `dl_throughput_per_user_mbps_roll36_mean`, `dl_throughput_per_user_mbps_roll36_std`, `dl_throughput_per_user_mbps_roll36_max`, `dl_throughput_per_user_mbps_roll36_min`, `active_users_roll6_mean`, `active_users_roll6_std`, `active_users_roll6_max`, `active_users_roll6_min`, `active_users_roll12_mean`, `active_users_roll12_std`, `active_users_roll12_max`, `active_users_roll12_min`, `active_users_roll36_mean`, `active_users_roll36_std`, `active_users_roll36_max`, `active_users_roll36_min`

### ewm
`bandwidth_utilization_pct_ewm3`, `bandwidth_utilization_pct_ewm12`, `latency_ms_ewm3`, `latency_ms_ewm12`, `packet_loss_pct_ewm3`, `packet_loss_pct_ewm12`, `dl_throughput_per_user_mbps_ewm3`, `dl_throughput_per_user_mbps_ewm12`, `active_users_ewm3`, `active_users_ewm12`

### congestion_history
`state_now`, `score_now`, `state_lag3`, `state_lag6`, `state_lag12`, `score_lag3`, `score_lag6`, `score_lag12`, `congested_frac_12`, `congested_frac_36`

### calendar_now
`hour`, `tod_sin`, `tod_cos`, `dow_sin`, `dow_cos`, `is_weekend`, `is_peak_hour`

### calendar_forecast_time
`fc_tod_sin`, `fc_tod_cos`, `fc_is_weekend`, `fc_is_peak_hour`

### site_static
`site_capacity_mbps`, `site_med_latency_ms`, `site_p90_util_pct`, `site_p90_users`

### site_relative
`latency_vs_site`, `util_vs_site_p90`, `users_vs_site_p90`

### region_onehot
`region__business_district`, `region__rural`, `region__suburban`, `region__urban_core`, `region__urban_residential`
