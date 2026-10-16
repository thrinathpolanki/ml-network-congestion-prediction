# Model comparison

> **SIMULATED DATA.** One generated dataset; seeds vary model initialisation/sampling only. Nothing here is real telecom performance.

## Process (fixed before looking at test results)

1. Every model trained with 4 seeds [42, 43, 44, 45]; early stopping on validation.
2. Operating points (calibration + F2 threshold) fitted on **validation** for every model alike.
3. Selection rule on validation only: keep **xgboost** unless a challenger beats it by >= 2.0 pooled seed-SDs on PR-AUC or on *operator area* (event recall averaged over 0.2-1.0 false alerts per site-day) without being clearly worse on the other. Needs >= 3 seeds.
4. Decision written to `models/production_model.json` at 2026-10-12T00:14:26+00:00 UTC, **before** the test set was scored.
5. Test set scored once (lock file `reports/test_results.json`).

## Validation evidence used for selection

| Model | PR-AUC (mean ± sd) | Operator area (mean ± sd) |
|---|---|---|
| xgboost | 0.8167 ± 0.0010 | 0.559 ± 0.019 |
| lstm | 0.8127 ± 0.0014 | 0.708 ± 0.022 |
| hybrid | 0.8193 ± 0.0007 | 0.652 ± 0.026 |

z-scores vs default (positive = challenger better):

- lstm: PR-AUC z = -3.28, operator-area z = 7.26
- hybrid: PR-AUC z = 2.63, operator-area z = 4.12

**Chosen by the rule: `hybrid`.** hybrid beats xgboost by >= 2.0 pooled seed-SDs on at least one validation criterion without being clearly worse on the other.

## Test results (mean ± sd over seeds; baselines are deterministic)

| Model | Accuracy | Macro-F1 | PR-AUC | Cong. recall | Cong. precision | New-onset recall | Event recall | False alerts / site-day |
|---|---|---|---|---|---|---|---|---|
| Majority class | 0.881 | 0.234 | 0.040 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Threshold rule (bandwidth) | 0.921 | 0.620 | 0.698 | 0.711 | 0.629 | 0.157 | 0.493 | 0.739 |
| Persistence (state now) | 0.927 | 0.614 | 0.700 | 0.659 | 0.660 | 0.000 | 0.390 | 0.483 |
| Seasonal naive (1d) | 0.881 | 0.442 | 0.319 | 0.436 | 0.357 | 0.379 | 0.459 | 0.978 |
| Seasonal naive (7d) | 0.899 | 0.467 | 0.424 | 0.452 | 0.457 | 0.359 | 0.452 | 0.761 |
| XGBoost (argmax) | 0.946 ± 0.000 | 0.673 ± 0.004 | 0.791 ± 0.002 | 0.665 ± 0.004 | 0.758 ± 0.002 | 0.269 ± 0.007 | 0.474 ± 0.003 | 0.382 ± 0.035 |
| XGBoost (tuned tau) | 0.933 ± 0.002 | 0.623 ± 0.012 | 0.791 ± 0.002 | 0.935 ± 0.012 | 0.526 ± 0.018 | 0.817 ± 0.029 | 0.894 ± 0.013 | 0.911 ± 0.046 |
| LSTM (argmax) | 0.937 ± 0.001 | 0.674 ± 0.004 | 0.788 ± 0.004 | 0.781 ± 0.031 | 0.702 ± 0.016 | 0.479 ± 0.045 | 0.656 ± 0.035 | 0.300 ± 0.032 |
| LSTM (tuned tau) | 0.926 ± 0.003 | 0.632 ± 0.011 | 0.788 ± 0.004 | 0.927 ± 0.015 | 0.537 ± 0.030 | 0.800 ± 0.041 | 0.887 ± 0.028 | 0.633 ± 0.083 |
| Hybrid (avg) (argmax) | 0.944 ± 0.000 | 0.682 ± 0.006 | 0.795 ± 0.002 | 0.729 ± 0.015 | 0.738 ± 0.006 | 0.379 ± 0.025 | 0.570 ± 0.017 | 0.296 ± 0.033 |
| Hybrid (avg) (tuned tau) | 0.933 ± 0.002 | 0.639 ± 0.013 | 0.795 ± 0.002 | 0.927 ± 0.016 | 0.546 ± 0.022 | 0.798 ± 0.040 | 0.885 ± 0.020 | 0.647 ± 0.048 |

95% CIs (day-block bootstrap, first seed): XGBoost (tuned tau): macro-F1 [0.559, 0.646], recall [0.905, 0.966]; LSTM (tuned tau): macro-F1 [0.565, 0.659], recall [0.908, 0.968]; Hybrid (avg) (tuned tau): macro-F1 [0.567, 0.662], recall [0.908, 0.967]

## Do the two models make the same mistakes? (tuned tau, first seed)

- Congested rows: 2057. Missed by XGBoost: 108; by LSTM: 107; by both: 84 (Jaccard of misses 0.64).
- False-alarm rows: XGBoost 1881, LSTM 1994, both 1694 (Jaccard 0.78).
- Caught by XGBoost 1949, by LSTM 1950, by either 1973.

## Cost, complexity and practicality

| | XGBoost | LSTM |
|---|---|---|
| Size | 1035 KB, 412 trees | 28,548 parameters |
| Inputs | 158 engineered features | 25 per step x seq + 13 static |
| History needed | 73 readings | 36 readings |
| Training time (mean, this machine) | 60 s | 198 s |
| End-to-end latency per site request (median / p95) | 49 / 64 ms | 48 / 72 ms |
| Interpretability | Built-in gain + TreeSHAP | None built in (post-hoc methods only) |
| Near-real-time suitability | Yes | Yes |

Both are dominated by feature construction (pandas), not by the model. Hybrid (average of both) would take 97 ms median.

## Known weaknesses

- CRITICAL class (tuned tau, first seed): XGBoost recall 0.27 / precision 0.50; LSTM recall 0.41 / precision 0.41 (support 222).
- Per-region results for regions with almost no congestion (rural, suburban) are not informative.
- The hybrid was scored on test.
- The selection rule compares each challenger only with the default, not with each other, and uses single-criterion z-scores; with 4 seeds the SD estimates are crude. See the narrative in the project report.
