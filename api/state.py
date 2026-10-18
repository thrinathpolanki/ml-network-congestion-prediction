"""In-process state: reading buffer, prediction log, metrics, and the application state holder."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.alerts.service import AlertService
from src.prediction.predictor import CongestionPredictor


class ReadingsStore:
    """Per-site rolling buffer of recent readings (de-duplicated by timestamp, newest wins)."""

    def __init__(self, max_per_site: int, freq: str):
        self._max, self._freq = int(max_per_site), freq
        self._d: dict[str, dict[pd.Timestamp, dict]] = defaultdict(dict)
        self._lock = threading.Lock()

    def _key(self, ts) -> pd.Timestamp:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return t.round(self._freq)

    def add(self, readings: list[dict]) -> set[str]:
        touched: set[str] = set()
        with self._lock:
            for r in readings:
                site = str(r["site_id"])
                self._d[site][self._key(r["timestamp"])] = dict(r)
                touched.add(site)
            for site in touched:
                buf = self._d[site]
                if len(buf) > self._max:
                    for k in sorted(buf)[: len(buf) - self._max]:
                        del buf[k]
        return touched

    def history_df(self, site: str) -> pd.DataFrame:
        with self._lock:
            buf = self._d.get(site, {})
            rows = [buf[k] for k in sorted(buf)]
        return pd.DataFrame(rows)

    def sites(self) -> list[str]:
        with self._lock:
            return sorted(self._d)

    def count(self, site: str) -> int:
        with self._lock:
            return len(self._d.get(site, {}))

    def last_reading(self, site: str) -> dict | None:
        with self._lock:
            buf = self._d.get(site)
            return dict(buf[max(buf)]) if buf else None

    def last_time(self, site: str) -> pd.Timestamp | None:
        with self._lock:
            buf = self._d.get(site)
            return max(buf) if buf else None


class PredictionLog:
    """Recent assessments per site; also keeps the latest one for /sites."""

    def __init__(self, per_site: int):
        self._d: dict[str, deque] = defaultdict(lambda: deque(maxlen=int(per_site)))
        self._lock = threading.Lock()

    def add(self, a: dict) -> bool:
        """Record unless an assessment for the same or a later time already exists. Returns True if recorded."""
        with self._lock:
            q = self._d[a["site_id"]]
            if q and q[-1]["prediction_time"] >= a["prediction_time"]:
                return False
            q.append(a)
            return True

    def latest(self, site: str) -> dict | None:
        with self._lock:
            q = self._d.get(site)
            return q[-1] if q else None

    def recent(self, site: str, limit: int) -> list[dict]:
        with self._lock:
            return list(self._d.get(site, []))[-limit:]


class Metrics:
    """Small thread-safe metrics registry (counters + request latency), JSON or Prometheus text."""

    def __init__(self):
        self.started = time.time()
        self._lock = threading.Lock()
        self._c: dict[tuple, int] = defaultdict(int)
        self._lat: deque = deque(maxlen=5000)

    def inc(self, name: str, n: int = 1, **labels) -> None:
        with self._lock:
            self._c[(name, tuple(sorted(labels.items())))] += n

    def observe_request(self, method: str, route: str, status: int, ms: float) -> None:
        self.inc("requests_total", method=method, route=route, status=str(status))
        with self._lock:
            self._lat.append(ms)

    def snapshot(self) -> dict:
        with self._lock:
            c, lat = dict(self._c), list(self._lat)
        by = lambda name, label: {dict(k[1]).get(label, ""): v for k, v in c.items() if k[0] == name}  # noqa: E731
        total = lambda name: sum(v for k, v in c.items() if k[0] == name)  # noqa: E731
        reqs = [(dict(k[1]), v) for k, v in c.items() if k[0] == "requests_total"]
        by_status: dict[str, int] = defaultdict(int)
        by_route: dict[str, int] = defaultdict(int)
        for lab, v in reqs:
            by_status[lab["status"]] += v
            by_route[f'{lab["method"]} {lab["route"]}'] += v
        return {"uptime_seconds": round(time.time() - self.started, 1),
                "requests": {"total": sum(by_status.values()), "by_status": dict(by_status), "by_route": dict(by_route)},
                "latency_ms": ({"count": len(lat), "avg": float(np.mean(lat)), "p50": float(np.percentile(lat, 50)),
                                "p95": float(np.percentile(lat, 95)), "max": float(np.max(lat))} if lat else {"count": 0}),
                "predictions": {"total": total("predictions_total"), "by_risk_level": by("predictions_total", "risk_level")},
                "alerts": {"total": total("alerts_total"), "by_severity": by("alerts_total", "severity"),
                           "by_kind": by("alerts_total", "kind")},
                "readings_ingested": total("readings_ingested_total"),
                "errors": {"total": total("errors_total"), "by_code": by("errors_total", "code")}}

    def prometheus(self) -> str:
        with self._lock:
            c, lat = dict(self._c), list(self._lat)
        lines = []
        for name in sorted({k[0] for k in c}):
            lines += [f"# TYPE ncp_{name} counter"]
            for (n, labels), v in sorted(c.items()):
                if n == name:
                    lab = ",".join(f'{k}="{str(val).replace(chr(34), "")}"' for k, val in labels)
                    lines.append(f"ncp_{name}{{{lab}}} {v}" if lab else f"ncp_{name} {v}")
        lines += ["# TYPE ncp_uptime_seconds gauge", f"ncp_uptime_seconds {time.time() - self.started:.1f}"]
        if lat:
            lines += ["# TYPE ncp_request_latency_ms summary"]
            for q in (0.5, 0.95):
                lines.append(f'ncp_request_latency_ms{{quantile="{q}"}} {np.percentile(lat, q * 100):.2f}')
            lines += [f"ncp_request_latency_ms_count {len(lat)}", f"ncp_request_latency_ms_sum {sum(lat):.2f}"]
        return "\n".join(lines) + "\n"


@dataclass
class AppState:
    cfg: dict
    readings: ReadingsStore
    log: PredictionLog
    metrics: Metrics
    alerts: AlertService
    predictor: CongestionPredictor | None = None
    model_error: str | None = None
    api_key: str | None = None
    started_at: float = field(default_factory=time.time)
