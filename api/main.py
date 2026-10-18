"""FastAPI application factory.

    uvicorn api.main:app --host 0.0.0.0 --port 8000        # or: python scripts/run_api.py
    open http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.errors import register_handlers
from api.routes import router, secured
from api.state import AppState, Metrics, PredictionLog, ReadingsStore
from src.alerts.service import AlertService
from src.config import PROJECT_ROOT, get_config
from src.prediction.predictor import CongestionPredictor
from src.utils.logging_config import setup_logging

logger = logging.getLogger("api")

DESCRIPTION = """
Predicts **network congestion before it happens** and raises operator alerts.

> **Prototype trained on SIMULATED data.** Probabilities, risk levels and metrics describe the simulation,
> not any real telecom network.

**Typical use**
1. `POST /predict` with the last ~3-6 h of readings of one cell (stateless, or `raise_alerts=false` for what-if), or
2. stream readings with `POST /readings` and let the service keep per-site history and raise alerts,
3. read `GET /alerts`, `GET /sites`, `GET /sites/{id}/history`.
"""


def create_app(cfg: dict | None = None, predictor: CongestionPredictor | None = None,
               alert_service: AlertService | None = None, models_dir=None, load_model: bool = True) -> FastAPI:
    cfg = cfg or get_config()
    ac = cfg["api"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(cfg)
        alerts = alert_service or AlertService.from_config(cfg)
        state = AppState(cfg=cfg, readings=ReadingsStore(ac["buffer_per_site"], cfg["data"]["sampling_freq"]),
                         log=PredictionLog(ac["prediction_log_per_site"]), metrics=Metrics(), alerts=alerts,
                         api_key=os.environ.get(ac["api_key_env"]) or None)
        app.state.ncp = state
        if predictor is not None:
            state.predictor = predictor
        elif load_model:
            try:
                state.predictor = CongestionPredictor.load(cfg, models_dir)
            except Exception as e:  # noqa: BLE001 - report via /health instead of crashing
                state.model_error = f"{type(e).__name__}: {e}"
                logger.error("Model could not be loaded: %s", state.model_error)
                if ac["fail_fast_on_model_error"]:
                    raise
        sp = cfg["alerts"].get("state_path")
        if sp and alert_service is None:
            path = PROJECT_ROOT / sp
            if alerts.load_state(path):
                logger.info("Restored alert state from %s", path)
        logger.info("API started: model=%s, alert channels=%s, API key protection %s", state.predictor.name if state.predictor else None,
                    alerts.dispatcher.names, "ENABLED" if state.api_key else "disabled")
        yield
        if sp and alert_service is None:
            alerts.save_state(PROJECT_ROOT / sp)
        logger.info("API stopped")

    app = FastAPI(title="AI-Based Network Congestion Prediction & Alert System", version=ac["version"],
                  description=DESCRIPTION, lifespan=lifespan,
                  openapi_tags=[{"name": "prediction"}, {"name": "alerts"}, {"name": "sites"}, {"name": "system"}])
    if ac["cors_origins"]:
        app.add_middleware(CORSMiddleware, allow_origins=ac["cors_origins"], allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        route = getattr(request.scope.get("route"), "path", "unmatched")
        state = getattr(request.app.state, "ncp", None)
        if state is not None:
            state.metrics.observe_request(request.method, route, response.status_code, ms)
        response.headers["X-Request-ID"] = rid
        logger.log(logging.WARNING if response.status_code >= 500 else logging.INFO, "%s %s -> %d (%.1f ms) rid=%s",
                   request.method, route, response.status_code, ms, rid)
        return response

    register_handlers(app)
    app.include_router(router)
    app.include_router(secured)
    return app


app = create_app()
