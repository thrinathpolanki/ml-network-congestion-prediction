"""Uniform, structured error responses. Internal details go to the log, never to the client."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.features.engineering import InsufficientHistoryError
from src.prediction.predictor import ModelConfigError

logger = logging.getLogger("api.errors")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def _body(request: Request, status: int, code: str, message: str, details=None) -> JSONResponse:
    state = getattr(request.app.state, "ncp", None)
    if state is not None:
        state.metrics.inc("errors_total", code=code)
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details},
                                                     "request_id": rid}, headers={"X-Request-ID": rid or ""})


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api(request: Request, exc: ApiError):
        return _body(request, exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]), "message": e["msg"], "type": e["type"]}
                   for e in exc.errors()][:25]                       # inputs are deliberately not echoed back
        return _body(request, 422, "VALIDATION_ERROR", "The request body or parameters are invalid.", details)

    @app.exception_handler(InsufficientHistoryError)
    async def _history(request: Request, exc: InsufficientHistoryError):
        state = getattr(request.app.state, "ncp", None)
        need = state.predictor.required_steps if state and state.predictor else None
        return _body(request, 422, "INSUFFICIENT_HISTORY", str(exc), {"required_readings": need})

    @app.exception_handler(ModelConfigError)
    async def _model(request: Request, exc: ModelConfigError):
        logger.error("Model configuration error: %s", exc)
        return _body(request, 503, "MODEL_UNAVAILABLE", "The prediction model is not usable; see server logs.")

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 401: "UNAUTHORIZED"}.get(exc.status_code, "HTTP_ERROR")
        return _body(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", None)
        logger.exception("Unhandled error rid=%s %s %s", rid, request.method, request.url.path)
        return _body(request, 500, "INTERNAL_ERROR", "An internal error occurred. Quote the request_id when reporting it.")
