"""Data-source abstraction: the interface between ANY data feed and the ML system.

To plug in real telecom data later, subclass DataSource and return a DataFrame
that satisfies src/data/schema.py. Nothing downstream needs to change.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.data.generator import generate_dataset
from src.data.schema import PROVENANCE_COLUMN, validate_schema

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Contract: load() -> DataFrame with the columns in schema.REQUIRED_COLUMNS."""

    @abstractmethod
    def load(self) -> pd.DataFrame: ...


class SyntheticSource(DataSource):
    """Generates simulated data in memory."""

    def __init__(self, cfg: dict, **overrides):
        self.cfg, self.overrides = cfg, overrides

    def load(self) -> pd.DataFrame:
        return generate_dataset(self.cfg, **self.overrides)


class CSVSource(DataSource):
    """Loads a CSV/Parquet export (e.g. from an OSS/NMS) with optional column renaming.

    column_map maps *your* column names to the schema names, e.g.
        {"cell": "site_id", "dl_prb_util": "bandwidth_utilization_pct"}
    """

    def __init__(self, path: str | Path, column_map: dict[str, str] | None = None,
                 provenance: str = "real"):
        self.path, self.column_map, self.provenance = Path(path), column_map or {}, provenance

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")
        df = pd.read_parquet(self.path) if self.path.suffix == ".parquet" else pd.read_csv(self.path)
        df = df.rename(columns=self.column_map)
        validate_schema(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if PROVENANCE_COLUMN not in df.columns:
            df[PROVENANCE_COLUMN] = self.provenance
        logger.info("Loaded %d rows from %s", len(df), self.path)
        return df
