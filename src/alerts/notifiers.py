"""Notification channels. Each is isolated: one failing channel never blocks the others or the engine.

Configuration = config['alerts']['notifiers'] (enabled flag, minimum severity) + environment variables
(destinations and credentials; never stored in code or config):

  webhook : ALERT_WEBHOOK_URL (http/https), optional ALERT_WEBHOOK_TOKEN (sent as a Bearer token)
  email   : SMTP_HOST, ALERT_EMAIL_TO (comma separated), ALERT_EMAIL_FROM (default SMTP_USER),
            optional SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD
  log     : always available

Security: credentials are never logged (they are masked in any error text), only the host of a webhook
URL is logged, and SMTP credentials are never sent over a connection that did not upgrade to TLS.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from email.message import EmailMessage
from typing import Callable, Mapping
from urllib.parse import urlparse

from src.alerts.engine import RANK, SEVERITIES, Alert

logger = logging.getLogger(__name__)


class Notifier:
    name = "notifier"
    min_severity = "INFO"
    secrets: tuple[str, ...] = ()

    def wants(self, alert: Alert) -> bool:
        return RANK[alert.severity] >= RANK[self.min_severity]

    def send(self, alert: Alert) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LogNotifier(Notifier):
    name = "log"

    def __init__(self, min_severity: str = "INFO"):
        self.min_severity = min_severity
        self._log = logging.getLogger("alerts")

    def send(self, alert: Alert) -> None:
        level = {"INFO": logging.INFO, "WARNING": logging.WARNING, "HIGH": logging.ERROR,
                 "CRITICAL": logging.CRITICAL}[alert.severity]
        self._log.log(level, "[%s] %s (%s, p=%.2f, %s)", alert.alert_id, alert.message, alert.kind,
                      alert.congestion_probability, alert.predicted_state)


def _default_post(url: str, body: bytes, headers: dict, timeout: float) -> int:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 (scheme validated at construction)
        return int(resp.status)


class WebhookNotifier(Notifier):
    name = "webhook"

    def __init__(self, url: str, token: str | None = None, timeout: float = 5.0, min_severity: str = "HIGH",
                 post: Callable[[str, bytes, dict, float], int] | None = None):
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.netloc:
            raise ValueError("ALERT_WEBHOOK_URL must be an http(s) URL")
        self.url, self.token, self.timeout, self.min_severity = url, token, float(timeout), min_severity
        self.host = u.hostname or "?"                                # only this is ever logged
        self._post = post or _default_post
        self.secrets = tuple(x for x in (url, token) if x)

    def send(self, alert: Alert) -> None:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        body = json.dumps({"service": "ai-network-congestion-prediction", "alert": alert.to_dict()},
                          default=str).encode("utf-8")
        status = self._post(self.url, body, headers, self.timeout)
        if not 200 <= int(status) < 300:
            raise RuntimeError(f"webhook {self.host} returned HTTP {status}")


class EmailNotifier(Notifier):
    name = "email"

    def __init__(self, host: str, to: list[str], sender: str, port: int = 587, user: str | None = None,
                 password: str | None = None, timeout: float = 10.0, min_severity: str = "CRITICAL",
                 smtp_factory: Callable[..., smtplib.SMTP] | None = None):
        if not (host and to and sender):
            raise ValueError("email needs host, recipients and sender")
        self.host, self.to, self.sender, self.port = host, to, sender, int(port)
        self.user, self.password, self.timeout, self.min_severity = user, password, float(timeout), min_severity
        self._factory = smtp_factory or smtplib.SMTP
        self.secrets = tuple(x for x in (password,) if x)      # a username is an identifier, not a secret

    def send(self, alert: Alert) -> None:
        msg = EmailMessage()
        msg["Subject"] = f"[{alert.severity}] Congestion risk {alert.site_id} ({alert.kind})"
        msg["From"], msg["To"] = self.sender, ", ".join(self.to)
        where = ""
        lat, lon = alert.context.get("latitude"), alert.context.get("longitude")
        if lat is not None and lon is not None:
            where = (f"Location: {lat:.6f},{lon:.6f}  Region: {alert.context.get('region', '?')}\n"
                     f"Map: https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=15/{lat:.6f}/{lon:.6f}\n")
        msg.set_content(f"{alert.message}\n\nRecommended action: {alert.recommended_action}\n{where}"
                        f"Prediction time: {alert.prediction_time}  Target time: {alert.target_time}\n"
                        f"Model: {alert.model}  Alert id: {alert.alert_id}\n")
        with self._factory(self.host, self.port, timeout=self.timeout) as smtp:
            smtp.ehlo()
            tls = smtp.has_extn("starttls")
            if tls:
                smtp.starttls()
                smtp.ehlo()
            if self.user and self.password:
                if not tls:
                    raise RuntimeError("refusing to send SMTP credentials without TLS")
                smtp.login(self.user, self.password)
            smtp.send_message(msg)


class NotifierDispatcher:
    def __init__(self, notifiers: list[Notifier]):
        self.notifiers = notifiers

    @property
    def names(self) -> list[str]:
        return [n.name for n in self.notifiers]

    def _mask(self, text: str, n: Notifier) -> str:
        for s in sorted(n.secrets, key=len, reverse=True):
            text = text.replace(s, "***")
        return text

    def dispatch(self, alert: Alert) -> dict[str, str]:
        """Send to every interested channel; returns {channel: 'sent' | 'skipped' | 'error: ...'}."""
        result: dict[str, str] = {}
        for n in self.notifiers:
            if not n.wants(alert):
                result[n.name] = "skipped (below min severity)"
                continue
            try:
                n.send(alert)
                result[n.name] = "sent"
            except Exception as e:  # noqa: BLE001 - isolation is the point
                msg = self._mask(f"{type(e).__name__}: {e}", n)
                result[n.name] = f"error: {msg}"
                logger.error("Notifier '%s' failed for %s: %s", n.name, alert.alert_id, msg)
        return result


def build_dispatcher(cfg: dict, env: Mapping[str, str] | None = None) -> NotifierDispatcher:
    """Create the channels that are both enabled in config and configured in the environment."""
    env = os.environ if env is None else env
    nc = cfg["alerts"]["notifiers"]
    out: list[Notifier] = []

    def minsev(key: str, default: str) -> str:
        s = str(nc.get(key, {}).get("min_severity", default)).upper()
        if s not in SEVERITIES:
            raise ValueError(f"Invalid min_severity '{s}' for {key}")
        return s

    if nc.get("log", {}).get("enabled", True):
        out.append(LogNotifier(minsev("log", "INFO")))
    if nc.get("webhook", {}).get("enabled", False):
        url = env.get("ALERT_WEBHOOK_URL", "").strip()
        if url:
            try:
                out.append(WebhookNotifier(url, env.get("ALERT_WEBHOOK_TOKEN") or None,
                                           nc["webhook"].get("timeout_seconds", 5), minsev("webhook", "HIGH")))
            except ValueError as e:
                logger.error("Webhook notifier disabled: %s", e)
        else:
            logger.info("Webhook notifier not configured (set ALERT_WEBHOOK_URL).")
    if nc.get("email", {}).get("enabled", False):
        host, to = env.get("SMTP_HOST", "").strip(), [x.strip() for x in env.get("ALERT_EMAIL_TO", "").split(",") if x.strip()]
        sender = env.get("ALERT_EMAIL_FROM") or env.get("SMTP_USER") or ""
        if host and to and sender:
            out.append(EmailNotifier(host, to, sender, int(env.get("SMTP_PORT") or 587), env.get("SMTP_USER") or None,
                                     env.get("SMTP_PASSWORD") or None, nc["email"].get("timeout_seconds", 10),
                                     minsev("email", "CRITICAL")))
        else:
            logger.info("Email notifier not configured (set SMTP_HOST, ALERT_EMAIL_TO, ALERT_EMAIL_FROM).")
    return NotifierDispatcher(out)
