# 🚀 Phase 12: Alert Engine

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added

| File | Purpose |
| --- | --- |
| `src/alerts/engine.py` | The per-site alert state machine |
| `src/alerts/store.py` | Thread-safe alert history, with optional JSONL persistence |
| `src/alerts/notifiers.py` | Log, webhook and email channels, isolated from each other |
| `src/alerts/service.py` | Engine + store + delivery behind one thread-safe call |
| `scripts/simulate_alerts.py` | Replays real predictions through the engine and measures the effect |
| `tests/test_alerts.py` | 33 new tests (165 total, all passing) |

Overwritten: `configs/config.yaml` (new `alerts:` section) and `.env.example`.

---

## ⚙️ Commands Executed

### Step 1: Run automated tests

```powershell
python -m pytest -q                       # expect: 165 passed
```

**Result:**

```text
.......................................................................................................................................................                                                                                                                                    [100%]
172 passed in 77.92s (0:01:17)
```

### Step 2: Run the alert simulation

```powershell
python scripts/simulate_alerts.py
```

**Result:**

```text
2026-10-23 22:33:58,255 | INFO     | src.prediction.predictor | Serving model from production_model.json: lstm
2026-10-23 22:33:58,271 | INFO     | src.prediction.predictor | Predictor ready: model=lstm, horizon=6 steps, tau=0.160, history=36 steps
2026-10-23 22:34:02,729 | INFO     | src.features.engineering | FeatureBuilder fitted on 241820 rows, 20 sites, 158 features
2026-10-23 22:34:11,122 | INFO     | src.features.dataset | Dataset built: {'train': 240999, 'val': 51700, 'test': 51840}
Model: lstm | alert config: confirm 2, cooldown 60 min, recovery 3 | SIMULATED DATA

=== TEST period: 51,840 predictions, 180 site-days, 146 congestion events ===
                                               scenario  notifications_per_site_day  event_coverage  n_events  median_lead_min  false_alert_episodes_per_site_day  alert_episodes  pages_excl_recovery_per_site_day
                    A. notify on EVERY risky prediction                      19.822           0.897       146           70.000                              0.600             212                               NaN
B. notify when risk starts (no debounce, no hysteresis)                       3.989           0.904       146           75.000                              0.589             212                               NaN
                             C. CONFIGURED alert engine                       3.400           0.870       146           70.000                              0.267             137                             2.639
Configured engine, notifications by kind: {'RAISED': 137, 'ESCALATED': 185, 'REMINDER': 153, 'DEESCALATED': 0, 'RECOVERED': 137}

Busiest cells (notifications excluding recovery notices) on the test period:
  CELL_006  urban_core         16.354309,80.522882     83  (9.2 per day)
  CELL_012  urban_core         16.332414,80.530624     76  (8.4 per day)
  CELL_016  urban_core         16.351801,80.526361     72  (8.0 per day)
  CELL_009  urban_core         16.351315,80.514270     65  (7.2 per day)
  CELL_020  business_district  16.337700,80.514420     46  (5.1 per day)
  share of all notifications from urban_core cells: 70% (5 of 20 cells)

=== Sensitivity on VALIDATION (tune here, not on test) ===
 confirm_steps  cooldown_min  notifications_per_site_day  event_coverage  median_lead_min  false_alert_episodes_per_site_day
             1            30                       6.367           0.953           80.000                              0.312
             1            60                       4.345           0.953           80.000                              0.312
             1           120                       3.482           0.953           80.000                              0.312
             2            30                       6.044           0.918           75.000                              0.240
             2            60                       4.089           0.918           75.000                              0.240
             2           120                       3.253           0.918           75.000                              0.240
             3            30                       5.754           0.906           70.000                              0.189
             3            60                       3.855           0.906           70.000                              0.189
             3           120                       3.036           0.906           70.000                              0.189

Saved ai-network-congestion-prediction\reports\alert_simulation.json
```
