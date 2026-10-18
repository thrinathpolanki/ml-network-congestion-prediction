"""HTTP endpoints. All handlers are synchronous (run in FastAPI's thread pool) because inference is CPU-bound."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from api.errors import ApiError
from api.schemas import (AlertsResponse, AssessmentOut, BatchPredictRequest, BatchPredictResponse, IngestRequest,
                         IngestResponse, PredictRequest, PredictResponse, SiteHistory, SiteInfo, SiteIngestResult,
                         SEVERITY)
from api.security import require_api_key
from api.state import AppState
from src.config import PROJECT_ROOT
from src.prediction.risk import RiskAssessment

logger = logging.getLogger("api")
router = APIRouter()
secured = APIRouter(dependencies=[Depends(require_api_key)])


def get_state(request: Request) -> AppState:
    return request.app.state.ncp


def need_predictor(state: AppState):
    if state.predictor is None:
        raise ApiError(503, "MODEL_UNAVAILABLE", "The prediction model is not loaded.",
                       {"reason": state.model_error or "unknown"})
    return state.predictor


def _frame(readings) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in readings])


def _record(state: AppState, a: RiskAssessment, alerts: list) -> list[dict]:
    """Log the assessment, count metrics and return alert dicts."""
    d = a.to_dict()
    state.log.add(d)
    state.metrics.inc("predictions_total", risk_level=a.risk_level)
    for al in alerts:
        state.metrics.inc("alerts_total", severity=al.severity, kind=al.kind)
    return [al.to_dict() for al in alerts]


# ------------------------------------------------------------------------ open endpoints
@router.get("/health", tags=["system"], summary="Liveness and readiness")
def health(state: AppState = Depends(get_state)):
    ok = state.predictor is not None
    body = {"status": "ok" if ok else "degraded", "model_loaded": ok, "model": state.predictor.name if ok else None,
            "version": state.cfg["api"]["version"], "uptime_seconds": round(time.time() - state.started_at, 1),
            "alert_channels": state.alerts.dispatcher.names,
            "problem": None if ok else state.model_error,
            "data_note": "Prototype trained on SIMULATED data."}
    return JSONResponse(body, status_code=200 if ok else 503)


# ----------------------------------------------------------------------- secured endpoints
@secured.get("/model/info", tags=["system"], summary="Which model is serving, and how it was evaluated")
def model_info(state: AppState = Depends(get_state)):
    info = need_predictor(state).info()
    perf = None
    summary = PROJECT_ROOT / state.cfg["paths"]["reports_dir"] / "test_results_summary.json"
    if summary.exists():
        try:
            rows = {r["model"]: r for r in json.loads(summary.read_text())["test_table"]}
            row = rows.get({"lstm": "LSTM (tuned tau)", "xgboost": "XGBoost (tuned tau)", "hybrid": "Hybrid (avg) (tuned tau)"}.get(info["model"], ""))
            if row:
                perf = {k: round(row[k], 4) for k in ("macro_f1", "pr_auc", "cong_recall", "cong_precision",
                                                       "recall_new_onset", "event_recall", "false_alerts_per_site_day")}
        except (OSError, ValueError, KeyError):
            logger.warning("Could not read evaluation summary", exc_info=True)
    return {**info, "n_features_xgboost": len(state.predictor.builder.feature_columns),
            "test_metrics": perf,
            "test_metrics_note": "Held-out test period of the SIMULATED dataset, mean of several training seeds. "
                                 "Not a statement about real network performance." if perf else None}


@secured.post("/predict", response_model=PredictResponse, tags=["prediction"],
              summary="Congestion risk for ONE site from its recent readings")
def predict(req: PredictRequest, raise_alerts: bool = Query(True, description="Run the alert engine (stateful). "
                                                                              "False = pure what-if prediction."),
            state: AppState = Depends(get_state)):
    pred = need_predictor(state)
    if len({r.site_id for r in req.readings}) != 1:
        raise ApiError(422, "MULTIPLE_SITES", "/predict takes readings for exactly one site; use /predict/batch.")
    a = pred.predict_site(_frame(req.readings))
    alerts = []
    if raise_alerts:
        alerts = _record(state, a, state.alerts.handle(a))
    return {"assessment": a.to_dict(), "alerts": alerts, "alerts_evaluated": raise_alerts}


@secured.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"],
              summary="Congestion risk for several sites; one bad site never blocks the others")
def predict_batch(req: BatchPredictRequest, raise_alerts: bool = Query(True), state: AppState = Depends(get_state)):
    pred = need_predictor(state)
    results, errors = pred.predict_many(_frame(req.readings))
    out = []
    for a in sorted(results, key=lambda x: x.prediction_time):
        alerts = _record(state, a, state.alerts.handle(a)) if raise_alerts else []
        out.append({"assessment": a.to_dict(), "alerts": alerts, "alerts_evaluated": raise_alerts})
    return {"results": out, "errors": errors}


@secured.post("/readings", response_model=IngestResponse, tags=["prediction"],
              summary="Streaming mode: add readings to the per-site buffer and (optionally) predict")
def ingest(req: IngestRequest, state: AppState = Depends(get_state)):
    if len(req.readings) > state.cfg["api"]["max_readings_per_request"]:
        raise ApiError(413, "PAYLOAD_TOO_LARGE", f"At most {state.cfg['api']['max_readings_per_request']} readings per request.")
    touched = state.readings.add([r.model_dump() for r in req.readings])
    state.metrics.inc("readings_ingested_total", len(req.readings))
    pred, sites = state.predictor, {}
    for site in sorted(touched):
        n = state.readings.count(site)
        if not req.predict:
            sites[site] = {"status": "not_requested", "buffered_readings": n}
            continue
        if pred is None:
            sites[site] = {"status": "model_unavailable", "buffered_readings": n}
            continue
        hist = state.readings.history_df(site)
        if n < pred.required_steps:
            sites[site] = {"status": "insufficient_history", "buffered_readings": n, "required_readings": pred.required_steps}
            continue
        a = pred.predict_site(hist.tail(max(pred.required_steps + 30, 100)))
        if not state.log.add(a.to_dict()):                       # nothing newer than what was already assessed
            sites[site] = {"status": "unchanged", "buffered_readings": n, "assessment": state.log.latest(site)}
            continue
        alerts = state.alerts.handle(a)
        state.metrics.inc("predictions_total", risk_level=a.risk_level)
        for al in alerts:
            state.metrics.inc("alerts_total", severity=al.severity, kind=al.kind)
        sites[site] = {"status": "predicted", "buffered_readings": n, "assessment": a.to_dict(),
                       "alerts": [x.to_dict() for x in alerts]}
    return {"ingested": len(req.readings), "sites": sites}


@secured.get("/alerts", response_model=AlertsResponse, tags=["alerts"], summary="Alert history and currently active sites")
def list_alerts(site_id: Optional[str] = None, min_severity: Optional[SEVERITY] = None,
                kind: Optional[Literal["RAISED", "ESCALATED", "REMINDER", "DEESCALATED", "RECOVERED"]] = None,
                since: Optional[datetime] = Query(None, description="Only alerts whose prediction_time >= this"),
                limit: int = Query(100, ge=1, le=1000), state: AppState = Depends(get_state)):
    items = state.alerts.store.list(site_id, min_severity, kind, since.isoformat() if since else None, limit)
    st = state.alerts.status()
    return {"count": len(items), "alerts": [a.to_dict() for a in items], "active_sites": st["active_sites"],
            "suppressed": st["suppressed"]}


@secured.post("/alerts/{alert_id}/ack", tags=["alerts"], summary="Mark an alert as acknowledged")
def acknowledge(alert_id: str, state: AppState = Depends(get_state)):
    if not state.alerts.store.acknowledge(alert_id):
        raise ApiError(404, "ALERT_NOT_FOUND", f"No alert with id {alert_id} in memory.")
    return {"alert_id": alert_id, "acknowledged": True}


@secured.get("/sites", response_model=list[SiteInfo], tags=["sites"],
             summary="Known sites with their latest prediction and alert status")
def sites(state: AppState = Depends(get_state)):
    regions, trained = {}, set()
    if state.predictor is not None:
        bl = state.predictor.builder.baselines
        trained = set(bl.table.index) if bl.table is not None else set()
        regions = getattr(bl, "site_region", {}) or {}
    active = state.alerts.status()["active_sites"]
    out = []
    for s in sorted(trained | set(state.readings.sites())):
        last = state.readings.last_time(s)
        lastr = state.readings.last_reading(s) or {}
        out.append({"site_id": s, "region": regions.get(s) or lastr.get("region"),
                    "latitude": lastr.get("latitude"), "longitude": lastr.get("longitude"),
                    "known_from_training": s in trained,
                    "buffered_readings": state.readings.count(s),
                    "last_reading_time": None if last is None else last.isoformat(),
                    "latest_prediction": state.log.latest(s),
                    "active_alert_severity": active.get(s, {}).get("severity")})
    return out


@secured.get("/sites/{site_id}/history", response_model=SiteHistory, tags=["sites"],
             summary="Recent readings and predictions of one site (for charts)")
def site_history(site_id: str, limit: int = Query(288, ge=1, le=2000), state: AppState = Depends(get_state)):
    df = state.readings.history_df(site_id)
    preds = state.log.recent(site_id, limit)
    if df.empty and not preds:
        raise ApiError(404, "SITE_NOT_FOUND", f"No readings or predictions stored for site {site_id}.")
    rows = []
    if not df.empty:
        df = df.tail(limit).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        rows = json.loads(df.to_json(orient="records"))
    return {"site_id": site_id, "readings": rows, "predictions": preds}


@secured.get("/metrics", tags=["system"], summary="Operational metrics (JSON, or Prometheus text with ?format=prometheus)")
def metrics(format: Literal["json", "prometheus"] = "json", state: AppState = Depends(get_state)):
    if format == "prometheus":
        return PlainTextResponse(state.metrics.prometheus(), media_type="text/plain; version=0.0.4")
    snap = state.metrics.snapshot()
    snap["alert_engine"] = state.alerts.status()
    snap["model"] = state.predictor.name if state.predictor else None
    return snap
