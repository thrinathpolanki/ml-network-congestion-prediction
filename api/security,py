"""Optional API-key protection. Active only if the configured environment variable is set."""
from __future__ import annotations

import secrets

from fastapi import Header, Request

from api.errors import ApiError


def require_api_key(request: Request, x_api_key: str | None = Header(default=None, description="API key (only if the server has one configured)")) -> None:
    expected = request.app.state.ncp.api_key
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise ApiError(401, "UNAUTHORIZED", "A valid X-API-Key header is required.")
