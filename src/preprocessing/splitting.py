"""Chronological train/validation/test split with target purging.

Why never shuffle: a row at time t is strongly correlated with t-1 and t+1.
Random splitting puts near-identical neighbours in train and test, so the model
is "tested" on data it has effectively seen, which gives optimistic scores
that vanish in deployment. We always train on the past and test on the future.

Why purge: the target of a row at time t is the state at t + H. A training row
just before the boundary would have its target inside the validation period,
which leaks future information. We drop rows whose target time crosses into
the next period. Features are built only from the past, so a validation row may
legitimately use training-period history in its lags; that is not leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import horizon_steps


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    boundaries: dict
    purged_rows: dict

    def summary(self, target_col: str | None = "target_state_id") -> dict:
        out = {"boundaries": self.boundaries, "purged_rows": self.purged_rows}
        for name in ("train", "val", "test"):
            d = getattr(self, name)
            info = {"rows": int(len(d)),
                    "start": str(d["timestamp"].min()), "end": str(d["timestamp"].max())}
            if target_col and target_col in d.columns:
                info["target_distribution"] = d[target_col].value_counts(normalize=True).sort_index().round(4).to_dict()
            out[name] = info
        return out


def chronological_split(df: pd.DataFrame, cfg: dict, purge_steps: int | None = None,
                        boundaries: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> SplitResult:
    """Split by global time boundaries (same calendar cut for every site).

    `boundaries=(t1, t2)` reuses cut points computed earlier, so that artefacts
    fitted on 'the training period' stay consistent with the final split.
    """
    sc = cfg["preprocessing"]["split"]
    tf, vf = float(sc["train_frac"]), float(sc["val_frac"])
    if not (0 < tf < 1 and 0 < vf < 1 and tf + vf < 1):
        raise ValueError("Need 0 < train_frac, val_frac and train_frac + val_frac < 1.")
    if "timestamp" not in df.columns:
        raise KeyError("DataFrame needs a 'timestamp' column.")

    steps = horizon_steps(cfg) if purge_steps is None else int(purge_steps)
    horizon = pd.Timedelta(cfg["data"]["sampling_freq"]) * steps
    if boundaries is None:
        uniq = pd.Index(sorted(df["timestamp"].unique()))
        if len(uniq) < 10:
            raise ValueError("Too few distinct timestamps to split.")
        t1 = uniq[int(len(uniq) * tf)]
        t2 = uniq[int(len(uniq) * (tf + vf))]
    else:
        t1, t2 = pd.Timestamp(boundaries[0]), pd.Timestamp(boundaries[1])
        if not t1 < t2:
            raise ValueError("boundaries must satisfy t1 < t2.")

    ts = df["timestamp"]
    in_train, in_val, in_test = ts < t1, (ts >= t1) & (ts < t2), ts >= t2
    keep_train = in_train & (ts + horizon < t1)
    keep_val = in_val & (ts + horizon < t2)
    res = SplitResult(
        train=df.loc[keep_train].copy(), val=df.loc[keep_val].copy(), test=df.loc[in_test].copy(),
        boundaries={"train_end_exclusive": str(t1), "val_end_exclusive": str(t2),
                    "purge_steps": steps},
        purged_rows={"train": int((in_train & ~keep_train).sum()), "val": int((in_val & ~keep_val).sum())},
    )
    if min(len(res.train), len(res.val), len(res.test)) == 0:
        raise ValueError("A split is empty; use more data or different fractions.")
    return res
