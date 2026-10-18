"""Pydantic request/response models.

The `Reading` model is generated from src/data/schema.py so the API contract can never drift from the
data contract used by the ML pipeline: same 14 KPIs, same physical ranges. A KPI may be null (a missing
sensor value is handled downstream); a present value outside its physical range is rejected with a 422.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from src.data.schema import RAW_METRICS, VALID_RANGES

SEVERITY = Literal["INFO", "WARNING", "HIGH", "CRITICAL"]
RISK = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
STATE = Literal["NORMAL", "WARNING", "CONGESTED", "CRITICAL"]

READING_EXAMPLE = {
    "site_id": "CELL_014", "timestamp": "2026-02-01T18:00:00", "region": "urban_core",
    "latitude": 16.343976, "longitude": 80.524684, "location": "16.343976,80.524684", "site_capacity_mbps": 1000.0,
    "active_users": 540, "active_sessions": 980, "traffic_volume_gb": 24.6, "bandwidth_utilization_pct": 91.5,
    "download_throughput_mbps": 590.0, "upload_throughput_mbps": 104.0, "latency_ms": 86.0, "packet_loss_pct": 1.8,
    "jitter_ms": 12.5, "signal_quality_sinr_db": 11.2, "cpu_utilization_pct": 88.0, "dropped_call_rate_pct": 1.4,
    "handover_failure_rate_pct": 1.1, "retransmission_rate_pct": 6.2}


def _metric_fields() -> dict[str, Any]:
    return {m: (Optional[float], Field(default=None, ge=VALID_RANGES[m][0], le=VALID_RANGES[m][1],
                                       allow_inf_nan=False, description=f"{m} (valid range {VALID_RANGES[m]})"))
            for m in RAW_METRICS}


def _check_location(self):
    """If latitude, longitude and the combined 'lat,lon' string are all given they must agree."""
    if self.location is not None and self.latitude is not None and self.longitude is not None:
        if f"{self.latitude:.6f},{self.longitude:.6f}" != self.location:
            raise ValueError("location must equal 'latitude,longitude' (6 decimals)")
    return self


Reading = create_model(
    "Reading",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"example": READING_EXAMPLE}),
    __doc__="One 5-minute KPI reading for one cell/site.",
    site_id=(str, Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")),
    timestamp=(datetime, Field(..., description="End of the 5-minute interval (ISO 8601)")),
    region=(str, Field(..., min_length=1, max_length=64)),
    site_capacity_mbps=(float, Field(..., gt=0, le=100_000, allow_inf_nan=False)),
    latitude=(Optional[float], Field(None, ge=-90, le=90, allow_inf_nan=False, description="Cell position (optional metadata)")),
    longitude=(Optional[float], Field(None, ge=-180, le=180, allow_inf_nan=False)),
    location=(Optional[str], Field(None, pattern=r"^-?\d{1,2}\.\d{1,8},-?\d{1,3}\.\d{1,8}$", description='"lat,lon", e.g. "16.343976,80.524684"')),
    __validators__={"_location_consistent": model_validator(mode="after")(_check_location)},
    **_metric_fields(),
)


class PredictRequest(BaseModel):
    """Recent readings of ONE site (oldest to newest, any order accepted)."""
    model_config = ConfigDict(extra="forbid")
    readings: list[Reading] = Field(..., min_length=1, max_length=5000)


class BatchPredictRequest(BaseModel):
    """Recent readings of several sites in one flat list."""
    model_config = ConfigDict(extra="forbid")
    readings: list[Reading] = Field(..., min_length=1, max_length=20000)


class IngestRequest(BaseModel):
    """Stream readings into the per-site buffer; optionally predict right away."""
    model_config = ConfigDict(extra="forbid")
    readings: list[Reading] = Field(..., min_length=1, max_length=5000)
    predict: bool = True


class AssessmentOut(BaseModel):
    site_id: str
    prediction_time: str
    target_time: str
    prediction_horizon: str
    congestion_probability: float = Field(..., ge=0, le=1)
    risk_level: RISK
    predicted_state: STATE
    confidence: float = Field(..., ge=0, le=1)
    recommended_action: str
    class_probabilities: dict[str, float]
    model: str
    context: dict[str, Any] = {}


class AlertOut(BaseModel):
    alert_id: str
    site_id: str
    severity: SEVERITY
    kind: Literal["RAISED", "ESCALATED", "REMINDER", "DEESCALATED", "RECOVERED"]
    message: str
    prediction_time: str
    target_time: str
    congestion_probability: float
    risk_level: RISK
    predicted_state: str
    recommended_action: str
    prediction_horizon: str
    model: str = ""
    first_raised_at: Optional[str] = None
    duration_minutes: Optional[float] = None
    peak_severity: Optional[str] = None
    acknowledged: bool = False
    context: dict[str, Any] = {}


class PredictResponse(BaseModel):
    assessment: AssessmentOut
    alerts: list[AlertOut] = []
    alerts_evaluated: bool = Field(..., description="False when raise_alerts=false (pure what-if prediction)")


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    errors: dict[str, str] = {}


class SiteIngestResult(BaseModel):
    status: Literal["predicted", "unchanged", "insufficient_history", "model_unavailable", "not_requested"]
    buffered_readings: int
    required_readings: Optional[int] = None
    assessment: Optional[AssessmentOut] = None
    alerts: list[AlertOut] = []


class IngestResponse(BaseModel):
    ingested: int
    sites: dict[str, SiteIngestResult]


class AlertsResponse(BaseModel):
    count: int
    alerts: list[AlertOut]
    active_sites: dict[str, dict]
    suppressed: dict[str, int]


class SiteInfo(BaseModel):
    site_id: str
    region: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    known_from_training: bool
    buffered_readings: int
    last_reading_time: Optional[str] = None
    latest_prediction: Optional[AssessmentOut] = None
    active_alert_severity: Optional[SEVERITY] = None


class SiteHistory(BaseModel):
    site_id: str
    readings: list[dict[str, Any]]
    predictions: list[AssessmentOut]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: Optional[str] = None
