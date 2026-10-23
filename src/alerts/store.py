"""In-memory alert history (ring buffer) with optional append-only JSONL persistence."""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path

from src.alerts.engine import RANK, Alert

logger = logging.getLogger(__name__)


class AlertStore:
    def __init__(self, max_size: int = 1000, persist_path: str | Path | None = None):
        self._items: deque[Alert] = deque(maxlen=int(max_size))
        self._lock = threading.Lock()
        self._path = Path(persist_path) if persist_path else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, alert: Alert) -> None:
        with self._lock:
            self._items.append(alert)
            if self._path:
                try:
                    with open(self._path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(alert.to_dict(), default=str) + "\n")
                except OSError as e:                      # persistence must never break alerting
                    logger.error("Could not persist alert %s: %s", alert.alert_id, e)

    def list(self, site_id: str | None = None, min_severity: str | None = None, kind: str | None = None,
             since: str | None = None, limit: int = 100) -> list[Alert]:
        """Newest first."""
        with self._lock:
            items = list(self._items)
        out = []
        for a in reversed(items):
            if site_id and a.site_id != site_id:
                continue
            if min_severity and RANK[a.severity] < RANK[min_severity]:
                continue
            if kind and a.kind != kind:
                continue
            if since and a.prediction_time < since:
                continue
            out.append(a)
            if len(out) >= limit:
                break
        return out

    def acknowledge(self, alert_id: str) -> bool:
        with self._lock:
            for a in self._items:
                if a.alert_id == alert_id:
                    a.acknowledged = True
                    return True
        return False

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
