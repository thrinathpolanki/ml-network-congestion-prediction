import copy
import json
import logging
import threading

import numpy as np
import pandas as pd
import pytest

from src.alerts.engine import AlertEngine
from src.alerts.notifiers import (EmailNotifier, LogNotifier, NotifierDispatcher, WebhookNotifier,
                                  build_dispatcher)
from src.alerts.service import AlertService
from src.alerts.store import AlertStore
from src.prediction.risk import RiskEngine

L, M, H, C = "LOW", "MEDIUM", "HIGH", "CRITICAL"


def run(engine, seq, site="CELL_001", start="2026-02-01 10:00", step=5):
    out = []
    for i, (lvl, p) in enumerate(seq):
        t = pd.Timestamp(start) + pd.Timedelta(minutes=step * i)
        out.append(engine.evaluate(site, t, lvl, p, "CONGESTED" if lvl != L else "NORMAL",
                                   (t + pd.Timedelta(minutes=30)).isoformat(), "do something", "30_minutes", "lstm"))
    return out


def kinds(res):
    return [a.kind for r in res for a in r]


def P(lvl):
    return {L: 0.02, M: 0.30, H: 0.55, C: 0.85}[lvl]


def seq(*levels):
    return [(lv, P(lv)) for lv in levels]


@pytest.fixture
def eng(cfg):
    return AlertEngine(cfg)


def cfg_with(cfg, **kw):
    c = copy.deepcopy(cfg)
    c["alerts"].update(kw)
    return c


# ------------------------------- debounce ----------------------------------- #
def test_low_risk_never_alerts(eng):
    assert kinds(run(eng, seq(*[L] * 20))) == []


def test_single_glitchy_prediction_does_not_page_anyone(eng):
    assert kinds(run(eng, seq(L, M, L, L, L))) == []
    assert eng.suppressed["debounce"] == 1 and not eng.is_active("CELL_001")


def test_two_consecutive_risky_predictions_raise_warning(eng):
    res = run(eng, seq(M, M))
    assert kinds(res) == ["RAISED"] and res[0] == [] and res[1][0].severity == "WARNING"


def test_critical_bypasses_debounce_unless_disabled(cfg):
    assert kinds(run(AlertEngine(cfg), seq(C))) == ["RAISED"]
    off = AlertEngine(cfg_with(cfg, critical_bypasses_debounce=False))
    assert kinds(run(off, seq(C))) == [] and kinds(run(AlertEngine(cfg_with(cfg, critical_bypasses_debounce=False)), seq(C, C))) == ["RAISED"]


# ------------------------- cooldown / duplicate suppression ----------------- #
def test_unchanged_risk_is_not_repeated_within_cooldown_then_one_reminder(eng):
    res = run(eng, seq(*[M] * 20))                     # 5-min steps: raised at i=1, reminder due at i=13 (60 min later)
    assert kinds(res) == ["RAISED", "REMINDER"] and res[13][0].kind == "REMINDER"
    assert "persists" in res[13][0].message and res[13][0].duration_minutes == 60.0
    assert eng.suppressed["cooldown"] > 0


def test_reminders_can_be_disabled(cfg):
    assert kinds(run(AlertEngine(cfg_with(cfg, send_reminders=False)), seq(*[M] * 30))) == ["RAISED"]


def test_duplicate_and_out_of_order_inputs_are_ignored(eng):
    t = pd.Timestamp("2026-02-01 10:00")
    a = eng.evaluate("S", t, C, 0.9)
    assert len(a) == 1
    assert eng.evaluate("S", t, C, 0.9) == []                                   # same timestamp re-submitted
    assert eng.evaluate("S", t - pd.Timedelta(minutes=5), M, 0.3) == []         # older than last processed
    assert eng.suppressed["duplicate_input"] == 2


# ------------------------------- escalation --------------------------------- #
def test_escalation_is_immediate_even_inside_cooldown(eng):
    res = run(eng, seq(M, M, H))
    assert kinds(res) == ["RAISED", "ESCALATED"] and res[2][0].severity == "HIGH"
    assert "escalated" in res[2][0].message


def test_flapping_between_levels_does_not_renotify_within_cooldown(eng):
    res = run(eng, seq(M, M, H, M, H, M, H))           # HIGH was notified at i=2; bounces are silent
    assert kinds(res) == ["RAISED", "ESCALATED"]


def test_reescalation_after_cooldown_is_notified(eng):
    res = run(eng, seq(M, M, H, *[M] * 13, H))         # i=2 HIGH; i=16 (70 min later) HIGH again
    assert kinds(res).count("ESCALATED") == 2


def test_deescalation_is_silent_by_default_and_optional(cfg, eng):
    assert kinds(run(eng, seq(M, M, H, M))) == ["RAISED", "ESCALATED"] and eng.state["CELL_001"].active_severity == "WARNING"
    on = AlertEngine(cfg_with(cfg, notify_deescalation=True))
    assert kinds(run(on, seq(M, M, H, M))) == ["RAISED", "ESCALATED", "DEESCALATED"]


# -------------------------------- recovery ---------------------------------- #
def test_recovery_needs_consecutive_low_predictions(eng):
    res = run(eng, seq(M, M, L, L, M, L, L))           # the M at i=4 resets the clear counter
    assert kinds(res) == ["RAISED"] and eng.is_active("CELL_001")
    res2 = run(eng, seq(L), start="2026-02-01 11:00")          # clear counter was already at 2 -> this is the 3rd
    assert kinds(res2) == ["RECOVERED"]
    a = res2[0][0]
    assert a.severity == "INFO" and a.peak_severity == "WARNING" and a.duration_minutes > 0 and not eng.is_active("CELL_001")
    assert "returned to normal" in a.message


def test_recovered_reports_the_peak_severity(eng):
    res = run(eng, seq(M, M, C, L, L, L))
    assert res[-1][0].kind == "RECOVERED" and res[-1][0].peak_severity == "CRITICAL"


def test_rearm_holdoff_suppresses_quick_warning_but_not_high(eng, cfg):
    res = run(eng, seq(M, M, L, L, L, M, M, M))        # recovered at i=4 (t=20); WARNING may re-raise from t>=35 (i=7)
    assert kinds(res) == ["RAISED", "RECOVERED", "RAISED"] and res[7][0].kind == "RAISED" and eng.suppressed["rearm"] == 1
    e2 = AlertEngine(cfg)
    res2 = run(e2, seq(M, M, L, L, L, H, H))
    assert kinds(res2) == ["RAISED", "RECOVERED", "RAISED"] and res2[6][0].severity == "HIGH"


# ------------------------------ independence / format ----------------------- #
def test_sites_are_independent(eng):
    run(eng, seq(M, M), site="A")
    assert eng.is_active("A") and not eng.is_active("B")
    assert kinds(run(eng, seq(L, L), site="B")) == [] and set(eng.active_sites()) == {"A"}


def test_message_matches_the_specified_style(eng):
    res = run(eng, [(M, 0.40), (M, 0.40)], site="CELL_014")
    assert res[1][0].message == "WARNING: Elevated congestion probability predicted for CELL_014 within the next 30 minutes (probability 40%)."
    assert res[1][0].alert_id == "ALT-000001" and res[1][0].prediction_horizon == "30_minutes"


def test_process_accepts_risk_assessments(cfg, eng):
    r = RiskEngine(cfg, 0.17)
    out = []
    for i in range(2):
        a = r.assess("CELL_007", pd.Timestamp("2026-02-01 18:00") + pd.Timedelta(minutes=5 * i), 0.5, np.array([.2, .2, .3, .3]), 2, "lstm", {"k": 1})
        out += eng.process(a)
    assert len(out) == 1 and out[0].severity == "HIGH" and out[0].context == {"k": 1} and out[0].model == "lstm"


# -------------------------------- config checks ----------------------------- #
@pytest.mark.parametrize("change", [
    {"severity_map": {"MEDIUM": "INFO"}}, {"severity_map": {"EXTREME": "HIGH"}}, {"cooldown_minutes": -1},
    {"confirm_steps": 0}, {"recovery_steps": 0}])
def test_invalid_alert_config_is_rejected(cfg, change):
    with pytest.raises(ValueError):
        AlertEngine(cfg_with(cfg, **change))


def test_template_validation(cfg):
    bad = copy.deepcopy(cfg["alerts"]["messages"]); bad["RAISED"] = "{severity} for {nonsense}"
    with pytest.raises(ValueError):
        AlertEngine(cfg_with(cfg, messages=bad))
    missing = copy.deepcopy(cfg["alerts"]["messages"]); del missing["RECOVERED"]
    with pytest.raises(ValueError):
        AlertEngine(cfg_with(cfg, messages=missing))


# ----------------------------- state persistence ---------------------------- #
def test_state_survives_restart_without_realerting(cfg, eng, tmp_path):
    run(eng, seq(M, M))
    state = json.loads(json.dumps(eng.export_state()))             # through JSON, as on disk
    e2 = AlertEngine(cfg); e2.import_state(state)
    res = run(e2, seq(M, M, M), start="2026-02-01 10:10")           # continues the same alert; inside cooldown
    assert kinds(res) == [] and e2.is_active("CELL_001")
    res2 = run(e2, seq(C), start="2026-02-01 10:30")
    assert res2[0][0].alert_id == "ALT-000002"                      # counter continued


# ------------------------------------ store --------------------------------- #
def _alerts(eng, n):
    """n alerts on n different sites: odd -> CRITICAL (immediate), even -> HIGH (needs 2 predictions)."""
    out = []
    for i in range(n):
        t = pd.Timestamp("2026-02-01") + pd.Timedelta(minutes=i)
        lvl = C if i % 2 else H
        out += eng.evaluate(f"S{i}", t, lvl, 0.8)
        out += eng.evaluate(f"S{i}", t + pd.Timedelta(seconds=30), lvl, 0.8)
    return out


def test_store_filters_limit_ack_and_ring_buffer(cfg, eng):
    st = AlertStore(max_size=5)
    for a in _alerts(eng, 8):
        st.add(a)
    assert len(st) == 5 and st.list()[0].site_id == "S7"             # newest first, oldest dropped
    assert {a.severity for a in st.list(min_severity="CRITICAL")} == {"CRITICAL"}
    assert len(st.list(limit=2)) == 2 and [a.site_id for a in st.list(site_id="S6")] == ["S6"]
    aid = st.list()[0].alert_id
    assert st.acknowledge(aid) and st.list()[0].acknowledged and not st.acknowledge("ALT-999999")


def test_store_persists_jsonl_and_survives_write_errors(cfg, eng, tmp_path, caplog):
    p = tmp_path / "logs" / "alerts.jsonl"
    st = AlertStore(10, p)
    for a in _alerts(eng, 3):
        st.add(a)
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["alert_id"] == "ALT-000001"
    bad = AlertStore(10, tmp_path)                                    # a directory: cannot be opened for writing
    with caplog.at_level(logging.ERROR):
        bad.add(_alerts(AlertEngine(cfg), 1)[0])
    assert len(bad) == 1 and "Could not persist" in caplog.text


# --------------------------------- notifiers -------------------------------- #
def one_alert(eng, severity_level=C):
    return eng.evaluate("CELL_014", pd.Timestamp("2026-02-01 18:00"), severity_level, 0.8)[0]


def test_webhook_min_severity_payload_and_secret_hygiene(cfg, caplog):
    sent = []
    wh = WebhookNotifier("https://hooks.example.com/services/SECRET123", "tok-abc", min_severity="HIGH",
                         post=lambda u, b, h, t: sent.append((u, json.loads(b), h)) or 200)
    d = NotifierDispatcher([wh])
    low = AlertEngine(cfg).evaluate("S", pd.Timestamp("2026-02-01"), C, .9)[0]
    low.severity = "WARNING"
    with caplog.at_level(logging.DEBUG):
        assert d.dispatch(low)["webhook"].startswith("skipped") and not sent
        assert d.dispatch(one_alert(AlertEngine(cfg), C))["webhook"] == "sent"
    url, body, headers = sent[0]
    assert body["alert"]["site_id"] == "CELL_014" and headers["Authorization"] == "Bearer tok-abc"
    assert wh.host == "hooks.example.com"
    assert "SECRET123" not in caplog.text and "tok-abc" not in caplog.text


def test_failing_channel_is_isolated_and_secrets_masked(cfg, caplog):
    def boom(u, b, h, t):
        raise ConnectionError(f"cannot reach {u} with token tok-xyz")
    bad = WebhookNotifier("https://h.example.com/path/PRIVATE", "tok-xyz", post=boom, min_severity="INFO")
    logged = []

    class Spy(LogNotifier):
        def send(self, alert):
            logged.append(alert.alert_id)
    d = NotifierDispatcher([bad, Spy()])
    with caplog.at_level(logging.ERROR):
        res = d.dispatch(one_alert(AlertEngine(cfg), C))
    assert res["webhook"].startswith("error") and "tok-xyz" not in res["webhook"] and "PRIVATE" not in res["webhook"]
    assert res["log"] == "sent" and logged                            # the other channel still ran
    assert "tok-xyz" not in caplog.text and "PRIVATE" not in caplog.text


def test_webhook_http_error_and_url_validation(cfg):
    CFG = cfg
    with pytest.raises(ValueError):
        WebhookNotifier("ftp://example.com/x")
    with pytest.raises(ValueError):
        WebhookNotifier("not a url")
    wh = WebhookNotifier("https://h.example.com/x", min_severity="INFO", post=lambda *a: 500)
    assert "HTTP 500" in NotifierDispatcher([wh]).dispatch(one_alert(AlertEngine(CFG), C))["webhook"]


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=None, tls=True):
        self.host, self.port, self.tls, self.sent, self.logged_in = host, port, tls, [], None
        FakeSMTP.instances.append(self)

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def ehlo(self): pass
    def has_extn(self, name): return self.tls
    def starttls(self): pass
    def login(self, u, p): self.logged_in = (u, p)
    def send_message(self, m): self.sent.append(m)


def test_email_sends_with_tls_and_refuses_plaintext_credentials(cfg):
    a = one_alert(AlertEngine(cfg), C)
    ok = EmailNotifier("smtp.example.com", ["noc@example.com"], "alerts@example.com", user="u", password="pw-123",
                       min_severity="CRITICAL", smtp_factory=lambda h, p, timeout: FakeSMTP(h, p, timeout))
    assert NotifierDispatcher([ok]).dispatch(a)["email"] == "sent"
    smtp = FakeSMTP.instances[-1]
    assert smtp.logged_in == ("u", "pw-123") and "CELL_014" in smtp.sent[0]["Subject"] and smtp.sent[0]["To"] == "noc@example.com"
    plain = EmailNotifier("smtp.example.com", ["noc@example.com"], "a@x.com", user="u", password="pw-123",
                          min_severity="INFO", smtp_factory=lambda h, p, timeout: FakeSMTP(h, p, timeout, tls=False))
    res = NotifierDispatcher([plain]).dispatch(a)["email"]
    assert res.startswith("error") and "without TLS" in res and "pw-123" not in res   # username 'u' must not be masked
    with pytest.raises(ValueError):
        EmailNotifier("", [], "")


def test_build_dispatcher_uses_environment_only(cfg):
    assert build_dispatcher(cfg, env={}).names == ["log"]
    d = build_dispatcher(cfg, env={"ALERT_WEBHOOK_URL": "https://h.example.com/x", "SMTP_HOST": "smtp.example.com",
                                   "ALERT_EMAIL_TO": "a@x.com, b@x.com", "ALERT_EMAIL_FROM": "n@x.com"})
    assert d.names == ["log", "webhook", "email"] and d.notifiers[2].to == ["a@x.com", "b@x.com"]
    assert build_dispatcher(cfg, env={"ALERT_WEBHOOK_URL": "javascript:alert(1)"}).names == ["log"]
    off = copy.deepcopy(cfg); off["alerts"]["notifiers"]["webhook"]["enabled"] = False
    assert "webhook" not in build_dispatcher(off, env={"ALERT_WEBHOOK_URL": "https://h.example.com/x"}).names
    bad = copy.deepcopy(cfg); bad["alerts"]["notifiers"]["email"]["min_severity"] = "LOUD"
    with pytest.raises(ValueError):
        build_dispatcher(bad, env={"SMTP_HOST": "s", "ALERT_EMAIL_TO": "a@x.com", "ALERT_EMAIL_FROM": "n@x.com"})


# ------------------------------------ service ------------------------------- #
def test_service_stores_dispatches_and_reports_status(cfg, tmp_path):
    svc = AlertService.from_config(cfg_with(cfg, persist_path=str(tmp_path / "a.jsonl")), env={})
    r = RiskEngine(cfg, 0.17)
    got = []
    for i in range(3):
        a = r.assess("CELL_021", pd.Timestamp("2026-02-01 12:00") + pd.Timedelta(minutes=5 * i), 0.6, np.array([.1, .2, .4, .3]), 2, "lstm")
        got += svc.handle(a)
    assert [x.kind for x in got] == ["RAISED"] and len(svc.store) == 1
    assert svc.delivery_log[-1]["result"]["log"] == "sent"
    st = svc.status()
    assert "CELL_021" in st["active_sites"] and st["channels"] == ["log"]
    svc.save_state(tmp_path / "state.json")
    svc2 = AlertService.from_config(cfg_with(cfg, persist_path=None), env={})
    assert svc2.load_state(tmp_path / "state.json") and svc2.engine.is_active("CELL_021")
    assert not svc2.load_state(tmp_path / "missing.json")


def test_service_is_thread_safe_across_sites(cfg):
    svc = AlertService.from_config(cfg_with(cfg, persist_path=None), env={})
    r = RiskEngine(cfg, 0.17)

    def worker(site):
        for i in range(30):
            svc.handle(r.assess(site, pd.Timestamp("2026-02-01") + pd.Timedelta(minutes=5 * i), 0.5, np.array([.2, .2, .3, .3]), 2))
    ts = [threading.Thread(target=worker, args=(f"S{k}",)) for k in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    ids = [a.alert_id for a in svc.store.list(limit=1000)]
    assert len(ids) == len(set(ids)) and {a.site_id for a in svc.store.list(limit=1000)} == {f"S{k}" for k in range(8)}


# ------------------------------- updated data: alerts know WHERE the problem is ------------------------- #
def _located(eng_cfg, **ctx):
    eng = AlertEngine(eng_cfg)
    context = {"latitude": 16.343976, "longitude": 80.524684, "location": "16.343976,80.524684", "region": "urban_core", **ctx}
    return eng.evaluate("CELL_014", pd.Timestamp("2026-02-01 18:00"), C, 0.9, "CONGESTED", "", "act", "30_minutes", "hybrid", context)[0]


def test_alert_carries_the_cell_position_in_its_context(cfg):
    a = _located(cfg)
    assert a.context["location"] == "16.343976,80.524684" and json.dumps(a.to_dict())


def test_location_placeholder_in_message_templates(cfg):
    msgs = copy.deepcopy(cfg["alerts"]["messages"])
    msgs["RAISED"] = "{severity}: {site_id} at {location} - {phrase} within {horizon_minutes} min."
    a = _located(cfg_with(cfg, messages=msgs))
    assert a.message == "CRITICAL: CELL_014 at 16.343976,80.524684 - Very high congestion probability within 30 min."
    bare = AlertEngine(cfg_with(cfg, messages=msgs)).evaluate("S", pd.Timestamp("2026-02-01"), C, .9)[0]
    assert bare.message == "CRITICAL: S at  - Very high congestion probability within 30 min."     # unknown position -> empty, no crash
    assert "Location" not in cfg["alerts"]["messages"]["RAISED"]                                   # default wording unchanged


def test_email_body_shows_location_region_and_map_link_only_when_known(cfg):
    sent = []

    class Capture(FakeSMTP):
        def send_message(self, m):
            sent.append(m.get_content())

    mk = lambda: EmailNotifier("smtp.example.com", ["noc@example.com"], "a@x.com", min_severity="INFO",  # noqa: E731
                               smtp_factory=lambda h, p, timeout: Capture(h, p, timeout))
    assert NotifierDispatcher([mk()]).dispatch(_located(cfg))["email"] == "sent"
    body = sent[-1]
    assert "Location: 16.343976,80.524684  Region: urban_core" in body
    assert "https://www.openstreetmap.org/?mlat=16.343976&mlon=80.524684#map=15/16.343976/80.524684" in body
    plain = AlertEngine(cfg).evaluate("S", pd.Timestamp("2026-02-01"), C, .9)[0]
    NotifierDispatcher([mk()]).dispatch(plain)
    assert "Location:" not in sent[-1] and "openstreetmap" not in sent[-1]


def test_webhook_payload_includes_the_position(cfg):
    got = []
    wh = WebhookNotifier("https://h.example.com/x", min_severity="INFO", post=lambda u, b, h, t: got.append(json.loads(b)) or 200)
    NotifierDispatcher([wh]).dispatch(_located(cfg))
    assert got[0]["alert"]["context"]["latitude"] == pytest.approx(16.343976)
