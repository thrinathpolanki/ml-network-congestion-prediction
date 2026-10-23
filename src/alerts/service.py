"""AlertService: engine + store + delivery, behind one thread-safe call."""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Mapping

from src.alerts.engine import Alert, AlertEngine
from src.alerts.notifiers import NotifierDispatcher, build_dispatcher
from src.alerts.store import AlertStore
from src.config import PROJECT_ROOT
from src.prediction.risk import RiskAssessment

logger = logging.getLogger(__name__)


class AlertService:
    def __init__(self, engine: AlertEngine, store: AlertStore, dispatcher: NotifierDispatcher):
        self.engine, self.store, self.dispatcher = engine, store, dispatcher
        self._lock = threading.Lock()
        self.delivery_log: deque = deque(maxlen=200)

    @staticmethod
    def from_config(cfg: dict, env: Mapping[str, str] | None = None, persist: bool = True) -> "AlertService":
        a = cfg["alerts"]
        path = a.get("persist_path") if persist else None
        if path and not Path(path).is_absolute():
            path = PROJECT_ROOT / path
        svc = AlertService(AlertEngine(cfg), AlertStore(a["history_size"], path), build_dispatcher(cfg, env))
        logger.info("Alert service ready; channels: %s", svc.dispatcher.names)
        return svc

    def handle(self, assessment: RiskAssessment) -> list[Alert]:
        """Run one assessment through the engine; store and deliver any alerts it produces."""
        with self._lock:
            alerts = self.engine.process(assessment)
            for a in alerts:
                self.store.add(a)
        for a in alerts:                                  # delivery outside the lock: slow channels must not block others
            self.delivery_log.append({"alert_id": a.alert_id, "result": self.dispatcher.dispatch(a)})
        return alerts

    def handle_many(self, assessments: list[RiskAssessment]) -> list[Alert]:
        out: list[Alert] = []
        for a in sorted(assessments, key=lambda x: x.prediction_time):
            out.extend(self.handle(a))
        return out

    def status(self) -> dict:
        with self._lock:
            return {"active_sites": self.engine.active_sites(), "suppressed": dict(self.engine.suppressed),
                    "alerts_in_memory": len(self.store), "channels": self.dispatcher.names}

    def save_state(self, path: str | Path) -> None:
        with self._lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(self.engine.export_state(), indent=2))

    def load_state(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        with self._lock:
            self.engine.import_state(json.loads(p.read_text()))
        return True
