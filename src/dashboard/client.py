"""Thin API client + replay helpers for the dashboard (no Streamlit imports, so everything here is unit-testable)."""
from __future__ import annotations

import pandas as pd

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RISK_COLORS = {"LOW": "#2e9e5b", "MEDIUM": "#e0b100", "HIGH": "#e8742a", "CRITICAL": "#d62f2f"}


class ApiError(RuntimeError):
    def __init__(self, status: int, error: str, detail):
        super().__init__(f"{status} {error}: {detail}")
        self.status, self.error, self.detail = status, error, detail


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe records (NaN -> None, timestamps -> ISO strings)."""
    d = df.copy()
    if "timestamp" in d.columns:
        d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return d.astype(object).where(d.notna(), None).to_dict("records")


class ApiClient:
    def __init__(self, base_url: str = "http://localhost:8000", http=None, timeout: float = 60):
        import requests
        self.base, self.http, self.timeout = base_url.rstrip("/"), http or requests, timeout

    def _call(self, method: str, path: str, **kw):
        if hasattr(self.http, "Session"):                              # the real `requests` module (TestClient has no Session)
            kw.setdefault("timeout", self.timeout)
        try:
            r = getattr(self.http, method)(self.base + path, **kw)
        except Exception as e:                                       # connection refused, DNS, timeout ...
            raise ApiError(0, "unreachable", str(e)) from e
        if r.status_code >= 400:
            try:
                j = r.json()
                raise ApiError(r.status_code, j.get("error", "http_error"), j.get("detail"))
            except ValueError:
                raise ApiError(r.status_code, "http_error", r.text[:200])
        return r.json()

    def health(self): return self._call("get", "/health")
    def model_info(self): return self._call("get", "/model/info")
    def performance(self): return self._call("get", "/model/performance")
    def metrics(self): return self._call("get", "/metrics")
    def sites(self): return self._call("get", "/sites")
    def alerts(self, limit: int = 200, **params): return self._call("get", "/alerts", params={"limit": limit, **params})
    def predict_batch(self, df: pd.DataFrame, raise_alerts: bool = True):
        return self._call("post", "/predict/batch", json={"readings": records(df), "raise_alerts": raise_alerts})
    def history(self, df: pd.DataFrame, limit: int = 288):
        return self._call("post", f"/predict/history?limit={limit}", json={"readings": records(df)})


def window_at(df: pd.DataFrame, t, steps: int) -> pd.DataFrame:
    """The last `steps` readings of every site up to and INCLUDING time t (never anything later)."""
    d = df[pd.to_datetime(df["timestamp"]) <= pd.Timestamp(t)]
    return d.sort_values(["site_id", "timestamp"]).groupby("site_id", group_keys=False).tail(steps)


def timeline(df: pd.DataFrame, steps: int) -> list[pd.Timestamp]:
    """Timestamps at which every site has at least `steps` readings (i.e. where the model can score)."""
    ts = sorted(pd.to_datetime(df["timestamp"]).unique())
    return [pd.Timestamp(x) for x in ts[steps - 1:]]
