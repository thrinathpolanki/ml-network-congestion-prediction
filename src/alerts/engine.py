"""Alert engine: risk assessments -> a small number of meaningful operator alerts.

State machine per site (all decisions use the PREDICTION's timestamp, so history can be replayed):

  CLEAR --(confirm_steps consecutive risky predictions; CRITICAL may bypass)--> ACTIVE(severity)   [RAISED]
  ACTIVE --higher severity (and not already notified within the cooldown)-------> ACTIVE(higher)   [ESCALATED]
  ACTIVE --same severity for >= cooldown---------------------------------------> (still ACTIVE)     [REMINDER]
  ACTIVE --lower but risky-----------------------------------------------------> ACTIVE(lower)     [silent, or DEESCALATED]
  ACTIVE --recovery_steps consecutive LOW predictions--------------------------> CLEAR             [RECOVERED, INFO]

Anti-spam mechanisms: debounce (confirm_steps), cooldown + per-severity notification memory,
duplicate/out-of-order input suppression, recovery hysteresis, re-arm hold-off after recovery.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from string import Formatter

import pandas as pd

from src.prediction.risk import RISK_LEVELS, RiskAssessment

logger = logging.getLogger(__name__)
SEVERITIES = ["INFO", "WARNING", "HIGH", "CRITICAL"]
RANK = {s: i for i, s in enumerate(SEVERITIES)}
KINDS = ["RAISED", "ESCALATED", "REMINDER", "DEESCALATED", "RECOVERED"]
PLACEHOLDERS = {"severity", "phrase", "site_id", "horizon_minutes", "probability_pct", "duration_minutes", "peak_severity", "location"}


@dataclass
class Alert:
    alert_id: str
    site_id: str
    severity: str
    kind: str
    message: str
    prediction_time: str
    target_time: str
    congestion_probability: float
    risk_level: str
    predicted_state: str
    recommended_action: str
    prediction_horizon: str
    model: str = ""
    first_raised_at: str | None = None
    duration_minutes: float | None = None
    peak_severity: str | None = None
    acknowledged: bool = False
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SiteState:
    active_severity: str | None = None
    first_raised_at: pd.Timestamp | None = None
    last_notified_at: pd.Timestamp | None = None
    notified_at_severity: dict = field(default_factory=dict)      # severity -> last notification time
    pending: int = 0                                              # consecutive risky predictions while CLEAR
    clear_count: int = 0                                          # consecutive LOW predictions while ACTIVE
    last_processed: pd.Timestamp | None = None
    last_recovered_at: pd.Timestamp | None = None
    peak_severity: str | None = None


class _SafeDict(dict):
    def __missing__(self, key):  # pragma: no cover - guarded by validation at init
        return "{" + key + "}"


class AlertEngine:
    def __init__(self, cfg: dict):
        a = cfg["alerts"]
        self.severity_map = {k: v for k, v in a["severity_map"].items()}
        for risk, sev in self.severity_map.items():
            if risk not in RISK_LEVELS or sev not in SEVERITIES or sev == "INFO":
                raise ValueError(f"Invalid severity_map entry {risk}->{sev}")
        self.confirm_steps = int(a["confirm_steps"])
        self.critical_bypass = bool(a["critical_bypasses_debounce"])
        self.cooldown = pd.Timedelta(minutes=float(a["cooldown_minutes"]))
        self.send_reminders = bool(a["send_reminders"])
        self.recovery_steps = int(a["recovery_steps"])
        self.rearm = pd.Timedelta(minutes=float(a["rearm_minutes"]))
        self.notify_deesc = bool(a["notify_deescalation"])
        if self.confirm_steps < 1 or self.recovery_steps < 1:
            raise ValueError("confirm_steps and recovery_steps must be >= 1")
        if self.cooldown < pd.Timedelta(0) or self.rearm < pd.Timedelta(0):
            raise ValueError("cooldown_minutes and rearm_minutes must be >= 0")
        self.phrases, self.templates = dict(a["phrases"]), dict(a["messages"])
        for kind in KINDS:                                        # fail at startup, not at 3 a.m.
            if kind not in self.templates:
                raise ValueError(f"Missing message template for {kind}")
            unknown = {f for _, f, _, _ in Formatter().parse(self.templates[kind]) if f} - PLACEHOLDERS
            if unknown:
                raise ValueError(f"Template {kind} uses unknown placeholders {sorted(unknown)}; allowed: {sorted(PLACEHOLDERS)}")
            self._render(kind, "WARNING", "S", 30, 0.5, 10.0, "HIGH")
        self.horizon_minutes = float(cfg["prediction"]["horizon_minutes"])
        self.state: dict[str, SiteState] = {}
        self._counter = 0
        self.suppressed = {"debounce": 0, "duplicate_input": 0, "cooldown": 0, "rearm": 0}

    # ------------------------------------------------------------------ helpers
    def _render(self, kind, severity, site_id, horizon_min, p, duration, peak, location: str = "") -> str:
        try:
            return self.templates[kind].format_map(_SafeDict(
                severity=severity, phrase=self.phrases.get(severity, "Congestion risk"), site_id=site_id,
                horizon_minutes=int(horizon_min) if float(horizon_min).is_integer() else horizon_min,
                probability_pct=f"{p:.0%}", duration_minutes=int(round(duration)), peak_severity=peak, location=location))
        except (KeyError, ValueError, IndexError) as e:
            raise ValueError(f"Bad message template for {kind}: {e}") from e

    def _make(self, kind: str, severity: str, a: dict, st: SiteState, t: pd.Timestamp) -> Alert:
        self._counter += 1
        dur = (t - st.first_raised_at).total_seconds() / 60 if st.first_raised_at is not None else 0.0
        msg = self._render(kind, severity, a["site_id"], self.horizon_minutes, a["p"], dur, st.peak_severity,
                           str(a.get("context", {}).get("location", "")))
        return Alert(
            alert_id=f"ALT-{self._counter:06d}", site_id=a["site_id"], severity=severity, kind=kind, message=msg,
            prediction_time=t.isoformat(), target_time=a["target_time"], congestion_probability=round(float(a["p"]), 4),
            risk_level=a["risk_level"], predicted_state=a["predicted_state"], recommended_action=a["action"],
            prediction_horizon=a["horizon"], model=a.get("model", ""),
            first_raised_at=None if st.first_raised_at is None else st.first_raised_at.isoformat(),
            duration_minutes=round(dur, 1) if kind in ("REMINDER", "RECOVERED") else None,
            peak_severity=st.peak_severity if kind == "RECOVERED" else None, context=a.get("context", {}))

    def _notified(self, st: SiteState, sev: str, t: pd.Timestamp) -> None:
        st.last_notified_at = t
        st.notified_at_severity[sev] = t

    # --------------------------------------------------------------------- core
    def evaluate(self, site_id: str, t, risk_level: str, p: float, predicted_state: str = "",
                 target_time: str = "", action: str = "", horizon: str = "", model: str = "",
                 context: dict | None = None) -> list[Alert]:
        t = pd.Timestamp(t)
        st = self.state.setdefault(site_id, SiteState())
        if st.last_processed is not None and t <= st.last_processed:
            self.suppressed["duplicate_input"] += 1
            return []
        st.last_processed = t
        sev = self.severity_map.get(risk_level)
        a = {"site_id": site_id, "p": p, "risk_level": risk_level, "predicted_state": predicted_state,
             "target_time": target_time, "action": action, "horizon": horizon, "model": model, "context": context or {}}
        out: list[Alert] = []

        if st.active_severity is None:                                           # ---- CLEAR
            if sev is None:
                st.pending = 0
                return out
            st.pending += 1
            need = 1 if (sev == "CRITICAL" and self.critical_bypass) else self.confirm_steps
            if st.pending < need:
                self.suppressed["debounce"] += 1
                return out
            if (sev == "WARNING" and st.last_recovered_at is not None
                    and t - st.last_recovered_at < self.rearm):
                self.suppressed["rearm"] += 1
                return out
            st.active_severity, st.first_raised_at, st.peak_severity = sev, t, sev
            st.pending = st.clear_count = 0
            st.notified_at_severity = {}
            self._notified(st, sev, t)
            out.append(self._make("RAISED", sev, a, st, t))
            return out

        if sev is None:                                                          # ---- ACTIVE, risk gone
            st.clear_count += 1
            if st.clear_count >= self.recovery_steps:
                out.append(self._make("RECOVERED", "INFO", a, st, t))
                st.active_severity, st.last_recovered_at = None, t
                st.first_raised_at = st.last_notified_at = None
                st.pending = st.clear_count = 0
                st.notified_at_severity = {}
            return out

        st.clear_count = 0                                                       # ---- ACTIVE, still risky
        if RANK[sev] > RANK[st.peak_severity or "INFO"]:
            st.peak_severity = sev
        if RANK[sev] > RANK[st.active_severity]:
            last = st.notified_at_severity.get(sev)
            st.active_severity = sev
            if last is None or t - last >= self.cooldown:
                self._notified(st, sev, t)
                out.append(self._make("ESCALATED", sev, a, st, t))
            else:
                self.suppressed["cooldown"] += 1
        elif RANK[sev] < RANK[st.active_severity]:
            st.active_severity = sev
            if self.notify_deesc:
                self._notified(st, sev, t)
                out.append(self._make("DEESCALATED", sev, a, st, t))
        elif self.send_reminders and t - st.last_notified_at >= self.cooldown:
            self._notified(st, sev, t)
            out.append(self._make("REMINDER", sev, a, st, t))
        else:
            self.suppressed["cooldown"] += 1
        return out

    def process(self, assessment: RiskAssessment) -> list[Alert]:
        """Feed one risk assessment (e.g. from CongestionPredictor.predict_site)."""
        return self.evaluate(assessment.site_id, assessment.prediction_time, assessment.risk_level,
                             assessment.congestion_probability, assessment.predicted_state, assessment.target_time,
                             assessment.recommended_action, assessment.prediction_horizon, assessment.model,
                             assessment.context)

    # ------------------------------------------------------------------- status
    def is_active(self, site_id: str) -> bool:
        return site_id in self.state and self.state[site_id].active_severity is not None

    def active_sites(self) -> dict[str, dict]:
        return {s: {"severity": st.active_severity, "first_raised_at": st.first_raised_at.isoformat(),
                    "peak_severity": st.peak_severity}
                for s, st in self.state.items() if st.active_severity is not None}

    def export_state(self) -> dict:
        ts = lambda x: None if x is None else pd.Timestamp(x).isoformat()  # noqa: E731
        return {"counter": self._counter, "sites": {s: {
            "active_severity": st.active_severity, "first_raised_at": ts(st.first_raised_at),
            "last_notified_at": ts(st.last_notified_at), "last_processed": ts(st.last_processed),
            "last_recovered_at": ts(st.last_recovered_at), "peak_severity": st.peak_severity,
            "pending": st.pending, "clear_count": st.clear_count,
            "notified_at_severity": {k: ts(v) for k, v in st.notified_at_severity.items()}}
            for s, st in self.state.items()}}

    def import_state(self, data: dict) -> None:
        tp = lambda x: None if x is None else pd.Timestamp(x)  # noqa: E731
        self._counter = int(data.get("counter", 0))
        for s, d in data.get("sites", {}).items():
            self.state[s] = SiteState(
                d["active_severity"], tp(d["first_raised_at"]), tp(d["last_notified_at"]),
                {k: tp(v) for k, v in d["notified_at_severity"].items()}, int(d["pending"]), int(d["clear_count"]),
                tp(d["last_processed"]), tp(d["last_recovered_at"]), d["peak_severity"])
