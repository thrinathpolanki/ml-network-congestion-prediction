"""Feature transformer: standard scaling + one-hot encoding, fitted on TRAIN only.

One object is fitted once and saved with the model; the API loads the same
object, so inference applies exactly the training-time transformation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """Scale numeric columns and one-hot encode categorical columns.

    NaNs pass through unchanged (XGBoost handles them; the LSTM path imputes
    before scaling). Unknown categories at inference become all-zero rows.
    """

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str] | None = None):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols or [])
        self.scaler = StandardScaler()
        self.categories_: dict[str, list] = {}
        self._fitted = False

    @property
    def feature_names_out(self) -> list[str]:
        self._check()
        cats = [f"{c}__{v}" for c in self.categorical_cols for v in self.categories_[c]]
        return self.numeric_cols + cats

    def _check(self) -> None:
        if not self._fitted:
            raise NotFittedError("FeatureTransformer is not fitted. Call fit() on training data.")

    def _require(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.numeric_cols + self.categorical_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for transformation: {missing}")

    def fit(self, train_df: pd.DataFrame) -> "FeatureTransformer":
        self._require(train_df)
        self.scaler.fit(train_df[self.numeric_cols].astype(float))
        self.categories_ = {c: sorted(train_df[c].dropna().astype(str).unique())
                            for c in self.categorical_cols}
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check()
        self._require(df)
        num = pd.DataFrame(self.scaler.transform(df[self.numeric_cols].astype(float)),
                           columns=self.numeric_cols, index=df.index)
        cat_frames = []
        for c in self.categorical_cols:
            vals = df[c].astype(str)
            unknown = ~vals.isin(self.categories_[c]) & df[c].notna()
            if unknown.any():
                logger.warning("Unknown %s categories at inference: %s",
                               c, sorted(vals[unknown].unique()))
            cat_frames.append(pd.DataFrame(
                {f"{c}__{v}": (vals == v).astype(float) for v in self.categories_[c]},
                index=df.index))
        return pd.concat([num] + cat_frames, axis=1)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    def save(self, path: str | Path) -> Path:
        self._check()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "FeatureTransformer":
        obj = joblib.load(path)
        if not isinstance(obj, FeatureTransformer):
            raise TypeError(f"{path} does not contain a FeatureTransformer")
        return obj
