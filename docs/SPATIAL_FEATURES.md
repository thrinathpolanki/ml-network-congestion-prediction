# Should cell position be a model feature? (validation experiment)

> **SIMULATED DATA.** Validation period only; the test set was not touched. Script: `scripts/ablation_spatial.py` (needs the XGBoost code from Phase 8; part of the full project).

## Question
The data now carries each cell's latitude/longitude. The EDA showed that, in the simulation, location matters only through the cell's region. This experiment asks whether adding spatial information still improves prediction.

## Variants (same XGBoost parameters, same seeds, early stopping on validation)
- **A** the 158 production features
- **B** A + latitude, longitude, distance from the map centre
- **C** A + mean and max *current* utilisation of the 3 nearest other cells at the same timestamp (needs neighbours' latest readings at serving time)

## Validation PR-AUC over seeds [42, 43, 44]

| Variant | PR-AUC (mean +- sd) | Difference vs A (mean) | Difference per seed |
|---|---|---|---|
| A baseline (158 features) | 0.8158 +- 0.0004 | +0.0000 | +0.0000, +0.0000, +0.0000 |
| B + position (lat, lon, distance) | 0.8170 +- 0.0003 | +0.0011 | +0.0013, +0.0012, +0.0009 |
| C + neighbour load (3 nearest cells) | 0.8161 +- 0.0004 | +0.0003 | +0.0003, +0.0003, +0.0003 |

## Reading the result
- **C (neighbour load): +0.0003**, inside the seed-to-seed spread (sd 0.0004). Expected: the simulator has no coupling between cells, so neighbours carry no extra information.
- **B (static position): +0.0011**, positive in every seed, so it is detectable, but tiny (about a tenth of a percentage point). A plausible explanation (NOT verified here) is that coordinates simply identify each training cell, letting the model memorise per-cell load levels; that would not carry over to a cell it has never seen.

## Decision
Position stays **out** of the model inputs (descriptive metadata for maps and dashboards only), because:
1. the gain is negligible next to the differences that matter (for example operator-level false alerts);
2. a feature that may only identify training cells is a liability for new cells, and the pipeline already falls back gracefully for unknown sites;
3. neighbour features add real-time dependencies between cells (more failure modes) for no measured benefit here.

## Caveats and how to revisit
- These findings describe the **simulator**, in which cells do not interact. Real networks share backhaul and interfere with neighbours, so neighbour-load features may well help there.
- Re-run `scripts/ablation_spatial.py` on real data (it only needs `latitude`/`longitude` columns). Judge variants by the validation spread across seeds, and check neighbour availability at serving time before adopting C.
- To test generalisation to unseen cells, hold out whole cells from training (not done here).
