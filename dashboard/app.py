"""Streamlit dashboard.  Run:  streamlit run dashboard/app.py   (API must be running: uvicorn src.api.main:app)

The dashboard owns no ML logic: it replays a telemetry feed (sample CSV = SIMULATED data) into the API and
displays what the API returns.  Swap FEED_PATH for a real export to monitor real data (see docs, Phase 22).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dashboard.client import RISK_COLORS, RISK_ORDER, ApiClient, ApiError, timeline, window_at  # noqa: E402

FEED_PATH = Path(os.environ.get("DASHBOARD_FEED", ROOT / "data" / "sample" / "sample_network_traffic.csv"))
API_URL = os.environ.get("API_URL", "http://localhost:8000")
color_scale = alt.Scale(domain=RISK_ORDER, range=[RISK_COLORS[r] for r in RISK_ORDER])

st.set_page_config(page_title="Network Congestion Prediction", page_icon="📡", layout="wide")


@st.cache_data(show_spinner=False)
def load_feed(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.sort_values(["site_id", "timestamp"])


# ------------------------------------------------------------------ sidebar / connection
st.sidebar.title("📡 Congestion Predictor")
api_url = st.sidebar.text_input("API URL", API_URL)
api = ApiClient(api_url, timeout=60)
try:
    health, info = api.health(), api.model_info()
except ApiError as e:
    st.error(f"Cannot reach the API at {api_url}. Start it with `uvicorn src.api.main:app --port 8000`.\n\n{e}")
    st.stop()

if not FEED_PATH.exists():
    st.error(f"Telemetry feed not found: {FEED_PATH}")
    st.stop()
feed = load_feed(str(FEED_PATH))
need = int(info["history_required_steps"])
times = timeline(feed, need)

st.sidebar.success(f"API connected · model: **{info['model']}**")
st.sidebar.caption(f"Horizon {info['horizon_minutes']:.0f} min · history needed {info['history_required_minutes']:.0f} min · "
                   f"alert channels: {', '.join(info['alert_channels']) or 'none'}")
st.sidebar.warning(info["data_note"])

if "cursor" not in st.session_state:
    st.session_state.cursor = 0
    st.session_state.last = None


def step_to(i: int, raise_alerts: bool = True):
    """Send everything every site 'knows' at times[i] to the API."""
    st.session_state.cursor = max(0, min(i, len(times) - 1))
    t = times[st.session_state.cursor]
    st.session_state.last = api.predict_batch(window_at(feed, t, need), raise_alerts=raise_alerts)


st.sidebar.subheader("Live replay")
st.sidebar.caption("Replays the feed into the API one 5-minute step at a time. Alerts follow the real debounce/cooldown rules.")
c1, c2 = st.sidebar.columns(2)
n_adv = st.sidebar.select_slider("Steps per click", [1, 3, 6, 12, 36], value=6)
try:
    if c1.button("▶ Advance", width="stretch"):
        base = st.session_state.cursor
        if st.session_state.last is None:
            step_to(base)
        else:
            for k in range(1, n_adv + 1):                      # intermediate steps matter: the alert engine is stateful
                step_to(base + k)
    if c2.button("⟲ Score now", width="stretch"):
        step_to(st.session_state.cursor, raise_alerts=False)
except ApiError as e:
    st.error(f"API error while scoring: {e}")
jump = st.sidebar.slider("Position in feed", 0, len(times) - 1, st.session_state.cursor)
if jump != st.session_state.cursor and st.sidebar.button("Jump here (no alerts)"):
    try:
        step_to(jump, raise_alerts=False)
    except ApiError as e:
        st.error(str(e))

now = times[st.session_state.cursor]
st.title("AI-Based Network Congestion Prediction & Alert System")
st.caption(f"Feed time: **{now:%Y-%m-%d %H:%M}** · position {st.session_state.cursor + 1}/{len(times)} · "
           "data: SIMULATED telemetry replayed into the API")

if st.session_state.last is None:
    st.info("Press **▶ Advance** or **⟲ Score now** in the sidebar to send the first window of telemetry to the API.")

tab_over, tab_site, tab_alerts, tab_model, tab_about = st.tabs(
    ["Network overview", "Site detail", "Alerts", "Model performance", "How it works"])

# ------------------------------------------------------------------ overview
with tab_over:
    sites = pd.DataFrame(api.sites()["sites"])
    if sites.empty:
        st.write("No sites scored yet.")
    else:
        m = api.metrics()
        at_risk = int((sites["risk_level"] != "LOW").sum())
        k = st.columns(5)
        k[0].metric("Sites monitored", len(sites))
        k[1].metric("At risk (MEDIUM+)", at_risk)
        k[2].metric("CRITICAL", int((sites["risk_level"] == "CRITICAL").sum()))
        k[3].metric("Active alerts", len(m["alert_service"]["active_sites"]))
        k[4].metric("Avg API latency", f"{m['avg_prediction_latency_ms']} ms" if m["avg_prediction_latency_ms"] else "–")

        ctx = pd.json_normalize(sites["context"])
        view = pd.concat([sites[["site_id", "risk_level", "congestion_probability", "predicted_state", "alert_active"]],
                          ctx[["region", "bandwidth_utilization_pct", "latency_ms", "packet_loss_pct", "active_users"]]], axis=1)
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Predicted congestion probability by site")
            ch = alt.Chart(view).mark_bar().encode(
                x=alt.X("congestion_probability:Q", scale=alt.Scale(domain=[0, 1]), title=f"P(congestion in {info['horizon_minutes']:.0f} min)"),
                y=alt.Y("site_id:N", sort="-x", title=None),
                color=alt.Color("risk_level:N", scale=color_scale, legend=alt.Legend(title="Risk")),
                tooltip=list(view.columns)).properties(height=max(220, 22 * len(view)))
            st.altair_chart(ch, width="stretch")
        with right:
            st.subheader("Site status")
            st.dataframe(view.rename(columns={"congestion_probability": "P(cong)", "bandwidth_utilization_pct": "util %",
                                              "latency_ms": "lat ms", "packet_loss_pct": "loss %"}),
                         hide_index=True, width="stretch", height=max(220, 22 * len(view)),
                         column_config={"P(cong)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        if sites["context"].apply(lambda c: "latitude" in c).any():
            st.subheader("Map")
            mp = pd.DataFrame({"lat": ctx["latitude"], "lon": ctx["longitude"], "risk": sites["risk_level"],
                               "color": sites["risk_level"].map(RISK_COLORS)}).dropna()
            st.map(mp, latitude="lat", longitude="lon", color="color", size=400)

# ------------------------------------------------------------------ site detail
with tab_site:
    ids = sorted(feed["site_id"].unique())
    default = 0
    if st.session_state.last and st.session_state.last["predictions"]:
        top = max(st.session_state.last["predictions"], key=lambda p: p["congestion_probability"])["site_id"]
        default = ids.index(top)
    site = st.selectbox("Site", ids, index=default)
    hours = st.select_slider("Look-back", [3, 6, 12, 24, 48], value=24, format_func=lambda h: f"{h} h")
    g = feed[(feed["site_id"] == site) & (feed["timestamp"] <= now)]
    recent = g[g["timestamp"] > now - pd.Timedelta(hours=hours)]
    if recent.empty:
        st.write("No data in range.")
    else:
        a, b = st.columns(2)
        for col, (name, label) in zip([a, b, a, b], [("bandwidth_utilization_pct", "Bandwidth utilization (%)"),
                                                      ("latency_ms", "Latency (ms)"), ("packet_loss_pct", "Packet loss (%)"),
                                                      ("traffic_volume_gb", "Traffic volume (GB / 5 min)")]):
            with col:
                st.markdown(f"**{label}**")
                st.line_chart(recent.set_index("timestamp")[name], height=170)
        st.subheader("Prediction history (backtest up to the feed time; each point uses only past data)")
        try:
            hist = pd.DataFrame(api.history(g.tail(need + hours * 12), limit=hours * 12)["rows"])
            hist["prediction_time"] = pd.to_datetime(hist["prediction_time"])
            line = alt.Chart(hist).mark_line(color="#555").encode(x=alt.X("prediction_time:T", title="time"),
                                                                  y=alt.Y("congestion_probability:Q", scale=alt.Scale(domain=[0, 1])))
            pts = alt.Chart(hist).mark_circle(size=40).encode(x="prediction_time:T", y="congestion_probability:Q",
                                                              color=alt.Color("risk_level:N", scale=color_scale),
                                                              tooltip=["prediction_time", "congestion_probability", "risk_level", "predicted_state"])
            st.altair_chart(line + pts, width="stretch")
        except ApiError as e:
            st.info(f"Not enough history yet to backtest this site: {e.detail}")

# ------------------------------------------------------------------ alerts
with tab_alerts:
    f1, f2 = st.columns(2)
    sev = f1.selectbox("Minimum severity", ["(all)", "WARNING", "HIGH", "CRITICAL"])
    kinds = f2.multiselect("Kind", ["RAISED", "ESCALATED", "REMINDER", "DEESCALATED", "RECOVERED"])
    al = api.alerts(**({"min_severity": sev} if sev != "(all)" else {}))["alerts"]
    if kinds:
        al = [a for a in al if a["kind"] in kinds]
    if not al:
        st.write("No alerts yet. Advance the replay to see alerts appear.")
    else:
        df = pd.DataFrame(al)[["prediction_time", "site_id", "severity", "kind", "congestion_probability", "message", "recommended_action"]]
        st.dataframe(df, hide_index=True, width="stretch")
    st.caption(f"Suppression so far: {api.metrics()['alert_service']['suppressed']}")

# ------------------------------------------------------------------ model performance
with tab_model:
    try:
        perf = api.performance()
        st.warning(perf["data_note"])
        t = pd.DataFrame(perf["test_table"])
        cols = ["model", "accuracy", "macro_f1", "cong_precision", "cong_recall", "cong_f1", "pr_auc", "roc_auc", "event_recall", "false_alerts_per_site_day"]
        t = t[[c for c in cols if c in t.columns]]
        st.markdown(f"Held-out **test** results (mean over seeds {perf['seeds']}). Selected model: "
                    f"**{(perf.get('selection') or {}).get('chosen', '?')}** — chosen on validation data *before* test scoring.")
        st.dataframe(t, hide_index=True, width="stretch")
        metric = st.selectbox("Compare models on", [c for c in t.columns if c != "model"], index=min(5, len(t.columns) - 2))
        st.altair_chart(alt.Chart(t).mark_bar().encode(x=alt.X(f"{metric}:Q"), y=alt.Y("model:N", sort="-x", title=None)),
                        width="stretch")
    except ApiError as e:
        st.info(f"Performance report unavailable: {e.detail}")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(f"""
**Pipeline:** telemetry → API validation → causal cleaning (5-min grid, short gap fill) → feature engineering (same
`FeatureBuilder` as training) → model (**{info['model']}**) → congestion probability → risk level → alert engine
(debounce, cooldown, recovery) → this dashboard.

**Risk levels:** MEDIUM ≥ {info['risk_thresholds']['MEDIUM']}, HIGH ≥ {info['risk_thresholds']['HIGH']}, CRITICAL ≥ {info['risk_thresholds']['CRITICAL']}
(probability of congestion {info['horizon_minutes']:.0f} minutes ahead).

**Honest scope:** trained and evaluated on simulated data only. Results show the method works on data with these
patterns, not how it would perform on a real network.
""")
