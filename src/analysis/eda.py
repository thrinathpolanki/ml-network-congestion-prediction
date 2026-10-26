"""Exploratory data analysis: figures + a machine-readable summary.

Runs on the TRAINING period only, so nothing learned here (peak hours, which
KPIs look predictive, ...) is informed by validation/test data.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.config import horizon_steps  # noqa: E402
from src.features.labeling import STATE_NAMES, add_derived_indicators  # noqa: E402

logger = logging.getLogger(__name__)
PEAK_FRACTION = 0.7  # peak hour = weekday mean utilisation >= 70 % of the daily maximum
COLORS = ["#2e9e5b", "#e6b422", "#e67e22", "#c0392b"]  # normal..critical
KPIS = ["bandwidth_utilization_pct", "latency_ms", "packet_loss_pct", "download_throughput_mbps",
        "jitter_ms", "dropped_call_rate_pct", "retransmission_rate_pct",
        "handover_failure_rate_pct", "cpu_utilization_pct", "signal_quality_sinr_db",
        "active_users", "dl_throughput_per_user_mbps"]


def _save(fig, out: Path, name: str) -> str:
    path = out / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def _shade(ax, ts, mask):
    ax.fill_between(ts, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                    color="#c0392b", alpha=0.15, lw=0)


def fig_timeseries(df, out):
    """One busy site, the week with most congestion: the 5 core KPIs over time."""
    site = df.groupby("site_id")["congestion_state_id"].apply(lambda s: (s >= 2).mean()).idxmax()
    d = df[df.site_id == site].set_index("timestamp")
    weekly = (d["congestion_state_id"] >= 2).rolling("7D").sum()
    end = weekly.idxmax()
    w = d.loc[end - pd.Timedelta("7D"):end]
    cong = (w["congestion_state_id"] >= 2).to_numpy()
    panels = [("traffic_volume_gb", "Traffic (GB / 5 min)"), ("bandwidth_utilization_pct", "Utilization (%)"),
              ("latency_ms", "Latency (ms)"), ("packet_loss_pct", "Packet loss (%)"),
              ("download_throughput_mbps", "DL throughput (Mbps)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 11), sharex=True)
    for ax, (c, lab) in zip(axes, panels):
        ax.plot(w.index, w[c], lw=0.8, color="#2c3e50")
        _shade(ax, w.index, cong)
        ax.set_ylabel(lab)
    axes[0].set_title(f"{site} ({d['region'].iloc[0]}) - busiest week; red = Congested/Critical (SIMULATED DATA)")
    return _save(fig, out, "01_timeseries_kpis"), site


def fig_distribution(df, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    share = df["congestion_state_id"].value_counts(normalize=True).reindex(range(4)).fillna(0)
    ax[0].bar(STATE_NAMES, share.values * 100, color=COLORS)
    ax[0].set_ylabel("% of observations"); ax[0].set_title("Congestion state distribution")
    for i, v in enumerate(share.values):
        ax[0].text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center")
    ct = pd.crosstab(df["region"], df["congestion_state_id"], normalize="index").reindex(columns=range(4), fill_value=0) * 100
    ct.plot(kind="barh", stacked=True, ax=ax[1], color=COLORS, legend=False)
    ax[1].set_xlabel("% of observations"); ax[1].set_title("State mix by region")
    ax[1].legend(STATE_NAMES, loc="lower right", fontsize=8)
    return _save(fig, out, "02_congestion_distribution")


def fig_by_time(df, out):
    d = df.assign(hour=df.timestamp.dt.hour, dow=df.timestamp.dt.dayofweek, cong=(df.congestion_state_id >= 2))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    d.groupby("hour")["cong"].mean().mul(100).plot(kind="bar", ax=ax[0], color="#c0392b")
    ax[0].set_title("Congested/Critical rate by hour"); ax[0].set_ylabel("%")
    r = d.groupby("dow")["cong"].mean().mul(100)
    ax[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], r.values,
              color=["#2c3e50"] * 5 + ["#7f8c8d"] * 2)
    ax[1].set_title("Congested/Critical rate by day of week")
    core = d[d.region == "urban_core"].pivot_table(index="dow", columns="hour", values="cong", aggfunc="mean") * 100
    im = ax[2].imshow(core.values, aspect="auto", cmap="Reds")
    ax[2].set_yticks(range(7)); ax[2].set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax[2].set_xticks(range(0, 24, 3)); ax[2].set_xticklabels(range(0, 24, 3))
    ax[2].set_title("urban_core: congestion % by day x hour"); fig.colorbar(im, ax=ax[2])
    return _save(fig, out, "03_congestion_by_hour_day")


def fig_by_site(df, out):
    rate = (df.assign(c=df.congestion_state_id >= 2).groupby(["site_id", "region"])["c"].mean().mul(100)
            .reset_index().sort_values("c", ascending=False))
    cmap = {r: c for r, c in zip(sorted(rate.region.unique()), plt.cm.tab10.colors)}
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(rate.site_id, rate.c, color=[cmap[r] for r in rate.region])
    ax.set_ylabel("Congested/Critical %"); ax.set_title("Congestion rate by site (colour = region)")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()], labels=list(cmap), fontsize=8)
    return _save(fig, out, "04_congestion_by_site"), rate


def fig_correlation(df, out):
    cols = KPIS + ["congestion_score"]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im); ax.set_title("Spearman correlation")
    return _save(fig, out, "05_correlation"), corr


def indicator_strength(df, min_events: int = 50) -> pd.DataFrame:
    """ROC-AUC of each KPI at time t for congestion at t + H.

    auc_pooled      : all sites together (inflated by between-site differences,
                      e.g. busy urban cells vs quiet rural cells)
    auc_within_site : mean over sites with >= min_events future events; this is
                      the honest signal for "will THIS cell congest soon?"
    AUC < 0.5 means the KPI falls before congestion (e.g. throughput, SINR), so
    strength = |AUC - 0.5| * 2 is direction-agnostic. 'congestion_score' (the
    label's own ingredient) is shown for reference.
    """
    df = add_derived_indicators(df)
    cols = KPIS + ["congestion_score"]
    fut = df.target_binary
    within = {c: [] for c in cols}
    for _, g in df.groupby("site_id"):
        if g.target_binary.sum() >= min_events and g.target_binary.nunique() == 2:
            for c in cols:
                within[c].append(roc_auc_score(g.target_binary, g[c].fillna(g[c].median())))
    rows = []
    for c in cols:
        pooled = roc_auc_score(fut, df[c].fillna(df[c].median()))
        w = float(np.mean(within[c])) if within[c] else float("nan")
        rows.append({"indicator": c, "auc_pooled": pooled, "auc_within_site": w,
                     "strength_within_site": abs(w - 0.5) * 2, "sites_used": len(within[c])})
    return pd.DataFrame(rows).sort_values("strength_within_site", ascending=False).reset_index(drop=True)


def fig_peak_behavior(df, out):
    d = df.assign(hour=df.timestamp.dt.hour)
    wd = d[d.timestamp.dt.dayofweek < 5]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for reg, g in wd.groupby("region"):
        ax[0].plot(g.groupby("hour")["bandwidth_utilization_pct"].mean(), label=reg)
        ax[1].plot(g.groupby("hour")["latency_ms"].mean(), label=reg)
        ax[2].plot(g.groupby("hour")["packet_loss_pct"].mean(), label=reg)
    for a, t in zip(ax, ["Utilization (%)", "Latency (ms)", "Packet loss (%)"]):
        a.set_title(f"Weekday mean {t}"); a.set_xlabel("hour of day")
    ax[0].legend(fontsize=8)
    prof = wd.groupby("hour")["bandwidth_utilization_pct"].mean()
    peak = sorted(int(h) for h in prof[prof >= PEAK_FRACTION * prof.max()].index)
    return _save(fig, out, "06_peak_hour_behavior"), peak


def fig_feature_dists(df, out):
    cols = ["bandwidth_utilization_pct", "latency_ms", "packet_loss_pct", "jitter_ms",
            "dl_throughput_per_user_mbps", "cpu_utilization_pct"]
    cong = df.congestion_state_id >= 2
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), cols):
        hi = df[c].quantile(0.995)
        bins = np.linspace(df[c].min(), hi, 60)
        ax.hist(df.loc[~cong, c].clip(upper=hi), bins=bins, density=True, alpha=0.6, label="Normal/Warning", color="#2e9e5b")
        ax.hist(df.loc[cong, c].clip(upper=hi), bins=bins, density=True, alpha=0.6, label="Congested/Critical", color="#c0392b")
        ax.set_title(c); ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    return _save(fig, out, "07_feature_distributions")


def fig_temporal_structure(df, out, steps: int):
    """Autocorrelation, state transitions at the horizon, congestion episode lengths."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    acfs = []
    for _, g in df.groupby("site_id"):
        s = g["bandwidth_utilization_pct"].to_numpy()
        s = s - np.nanmean(s)
        acfs.append([np.nanmean(s[k:] * s[:len(s) - k]) / np.nanmean(s * s) for k in range(0, 49)])
    ax[0].plot(np.arange(49) * 5, np.mean(acfs, axis=0))
    ax[0].axvline(steps * 5, ls="--", color="grey"); ax[0].set_xlabel("lag (min)")
    ax[0].set_title("Utilization autocorrelation (dashed = horizon)")
    tm = pd.crosstab(df.congestion_state_id, df.target_state_id, normalize="index").reindex(index=range(4), columns=range(4), fill_value=0)
    im = ax[1].imshow(tm.values, cmap="Blues", vmin=0, vmax=1)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(STATE_NAMES, rotation=30, fontsize=8)
    ax[1].set_yticks(range(4)); ax[1].set_yticklabels(STATE_NAMES, fontsize=8)
    ax[1].set_xlabel("state at t + horizon"); ax[1].set_ylabel("state at t")
    for i in range(4):
        for j in range(4):
            ax[1].text(j, i, f"{tm.values[i, j]:.2f}", ha="center", va="center", color="black" if tm.values[i, j] < .6 else "white")
    ax[1].set_title("P(state at t+H | state at t)")
    lens = []
    for _, g in df.groupby("site_id"):
        c = (g["congestion_state_id"].to_numpy() >= 2).astype(int)
        edges = np.diff(np.r_[0, c, 0])
        lens += list((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) * 5)
    ax[2].hist(lens, bins=40, color="#c0392b"); ax[2].set_xlabel("minutes"); ax[2].set_title("Congestion episode duration")
    return _save(fig, out, "08_temporal_structure"), tm, np.array(lens)


def fig_onset(df, out, pre: int = 12, post: int = 6):
    """Average KPI trajectory around the start of a congestion episode (event study)."""
    cols = ["bandwidth_utilization_pct", "latency_ms", "packet_loss_pct"]
    acc = {c: [] for c in cols}
    for _, g in df.groupby("site_id"):
        c = (g["congestion_state_id"].to_numpy() >= 2)
        onset = np.flatnonzero(c[1:] & ~c[:-1]) + 1
        # keep only onsets after >= 12 clear steps (not flicker inside an episode)
        onset = [o for o in onset if o >= pre and o + post < len(c) and not c[o - pre:o].any()]
        for col in cols:
            v = g[col].to_numpy()
            acc[col] += [v[o - pre:o + post + 1] for o in onset]
    x = np.arange(-pre, post + 1) * 5
    summary = {"n_onsets": len(acc[cols[0]])}
    if summary["n_onsets"] == 0:  # nothing to plot (e.g. very small / quiet dataset)
        return "", summary
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col in zip(ax, cols):
        m = np.array(acc[col])
        a.plot(x, np.nanmean(m, axis=0), color="#c0392b"); a.axvline(0, ls="--", color="grey")
        a.fill_between(x, np.nanpercentile(m, 25, axis=0), np.nanpercentile(m, 75, axis=0), alpha=0.2, color="#c0392b")
        a.set_title(f"{col} around onset (mean, IQR)"); a.set_xlabel("minutes from onset")
        summary[col] = {"t-60min": float(np.nanmean(m[:, 0])), "t-30min": float(np.nanmean(m[:, pre - 6])),
                        "t-10min": float(np.nanmean(m[:, pre - 2])), "onset": float(np.nanmean(m[:, pre]))}
    return _save(fig, out, "09_onset_trajectory"), summary


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def spatial_stats(df: pd.DataFrame, cfg: dict) -> dict:
    """Does WHERE a cell is tell us anything beyond WHAT KIND of cell it is?

    With ~20 cells these numbers are indicative only. In the simulation position is generated from the
    region (denser regions sit nearer the centre) and cells do not influence each other, so any distance
    effect should vanish once region is controlled for. Real networks may differ (shared backhaul,
    neighbouring-cell interference, events), which is why the check is worth keeping.
    """
    loc = cfg["generator"]["location"]
    cells = (df.assign(c=df.congestion_state_id >= 2)
               .groupby("site_id").agg(region=("region", "first"), lat=("latitude", "first"),
                                       lon=("longitude", "first"), rate=("c", "mean")))
    cells["dist_km"] = _haversine_km(float(loc["center_lat"]), float(loc["center_lon"]), cells.lat, cells.lon)
    rho = lambda a, b: float(pd.Series(a).corr(pd.Series(b), method="spearman"))  # noqa: E731
    resid = lambda col: cells[col] - cells.groupby("region")[col].transform("mean")  # noqa: E731
    hourly = (df.assign(c=(df.congestion_state_id >= 2).astype(float), h=df.timestamp.dt.floor("h"))
                .groupby(["h", "site_id"])["c"].mean().unstack())
    corr = hourly.corr().fillna(0.0)
    pairs = []
    ids = list(cells.index)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            pairs.append({"same_region": cells.region[a] == cells.region[b], "region": cells.region[a], "corr": corr.loc[a, b],
                          "dist_km": float(_haversine_km(cells.lat[a], cells.lon[a], cells.lat[b], cells.lon[b]))})
    pr = pd.DataFrame(pairs)
    same = pr[pr.same_region].copy()
    # control for region: compare pairs only with other pairs of the SAME region (rural pairs never congest, urban-core
    # pairs sit close together, so pooled pairs would show a spurious distance effect)
    for col in ("corr", "dist_km"):
        same[col + "_res"] = same[col] - same.groupby("region")[col].transform("mean")
    return {
        "n_cells": int(len(cells)),
        "rate_vs_distance_spearman_pooled": rho(cells.rate, cells.dist_km),
        "rate_vs_distance_spearman_within_region": rho(resid("rate"), resid("dist_km")),
        "mean_hourly_congestion_corr_same_region_pairs": float(same["corr"].mean()) if len(same) else None,
        "mean_hourly_congestion_corr_different_region_pairs": float(pr[~pr.same_region]["corr"].mean()),
        "corr_vs_distance_spearman_same_region_pairs_pooled_confounded": rho(same["corr"], same["dist_km"]) if len(same) > 3 else None,
        "corr_vs_distance_spearman_same_region_pairs_region_controlled": rho(same["corr_res"], same["dist_km_res"]) if len(same) > 3 else None,
        "mean_hourly_corr_by_region": {r: float(g["corr"].mean()) for r, g in same.groupby("region")},
        "n_same_region_pairs": int(len(same)),
        "cells": cells.round(4).reset_index().to_dict(orient="records"),
    }


def fig_cell_map(df: pd.DataFrame, cfg: dict, out: Path, stats: dict):
    """Cells on a lon/lat plane (marker size = congestion rate) and congestion rate vs distance from the centre."""
    cells = pd.DataFrame(stats["cells"])
    regions = sorted(cells.region.unique())
    col = dict(zip(regions, plt.cm.tab10.colors))
    loc = cfg["generator"]["location"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    for r in regions:
        c = cells[cells.region == r]
        ax[0].scatter(c.lon, c.lat, s=40 + c.rate * 2500, color=col[r], alpha=0.75, edgecolor="k", linewidth=0.5, label=r)
    for _, c in cells.iterrows():
        ax[0].annotate(c.site_id.replace("CELL_", ""), (c.lon, c.lat), fontsize=6.5, ha="center", va="center")
    ax[0].scatter([loc["center_lon"]], [loc["center_lat"]], marker="*", s=220, color="black", zorder=5, label="map centre")
    ax[0].set_xlabel("longitude"); ax[0].set_ylabel("latitude"); ax[0].set_aspect(1 / np.cos(np.radians(loc["center_lat"])))
    ax[0].set_title("Cell positions (marker size = share of time Congested/Critical; SIMULATED)"); ax[0].legend(fontsize=7)
    for r in regions:
        c = cells[cells.region == r]
        ax[1].scatter(c.dist_km, c.rate * 100, s=55, color=col[r], edgecolor="k", linewidth=0.5, label=r)
    ax[1].set_xlabel("distance from map centre (km)"); ax[1].set_ylabel("Congested/Critical (% of time)")
    ax[1].set_title(f"Congestion vs distance: pooled rho {stats['rate_vs_distance_spearman_pooled']:.2f}, "
                    f"within region rho {stats['rate_vs_distance_spearman_within_region']:.2f}")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "09a_cell_map.png", dpi=110); plt.close(fig)
    return str(out / "09a_cell_map.png")


def fig_sample_heatmap(sample: pd.DataFrame, out: Path, day: str | None = None) -> str:
    """Cell x time-of-day utilisation heatmap of the clean sample (every cell, every 5-minute step)."""
    d = sample.copy()
    d["day"] = d.timestamp.dt.strftime("%Y-%m-%d")
    day = day or sorted(d.day.unique())[-1]
    d = d[d.day == day]
    first = d.groupby("site_id")[["region"]].first()
    order = first.reset_index().sort_values(["region", "site_id"]).site_id.tolist()
    piv = d.pivot_table(index="site_id", columns=d.timestamp.dt.strftime("%H:%M"), values="bandwidth_utilization_pct").reindex(order)
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{c}  {first.region[c]}" for c in order], fontsize=7)
    ax.set_xticks(range(0, piv.shape[1], 36)); ax.set_xticklabels(piv.columns[::36])
    ax.set_xlabel(f"time of day, {day} (288 five-minute steps)"); ax.set_title("Utilisation (%) per cell across the day: SIMULATED sample, complete grid")
    fig.colorbar(im, label="bandwidth utilisation (%)")
    fig.tight_layout(); path = out / "09b_sample_day_heatmap.png"; fig.savefig(path, dpi=110); plt.close(fig)
    return str(path)


def run_eda(df: pd.DataFrame, cfg: dict, out_dir: Path) -> dict:
    """df must contain labels + targets (training period). Returns the summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = add_derived_indicators(df)
    steps = horizon_steps(cfg)
    s: dict = {"rows": int(len(df)), "period": [str(df.timestamp.min()), str(df.timestamp.max())],
               "data_note": "SIMULATED DATA - training period only"}
    s["state_share_pct"] = (df.congestion_state_id.value_counts(normalize=True).sort_index() * 100).round(2).to_dict()
    figs = {}
    figs["timeseries"], site = fig_timeseries(df, out_dir); s["timeseries_site"] = site
    figs["distribution"] = fig_distribution(df, out_dir)
    figs["by_time"] = fig_by_time(df, out_dir)
    d = df.assign(c=df.congestion_state_id >= 2, hour=df.timestamp.dt.hour)
    s["congested_rate_pct_by_region"] = (d.groupby("region")["c"].mean() * 100).round(2).to_dict()
    s["congested_rate_pct_weekday_vs_weekend"] = {
        "weekday": round(float(d[d.timestamp.dt.dayofweek < 5]["c"].mean() * 100), 2),
        "weekend": round(float(d[d.timestamp.dt.dayofweek >= 5]["c"].mean() * 100), 2)}
    hr = (d.groupby("hour")["c"].mean() * 100).round(2)
    s["top_congestion_hours"] = hr.sort_values(ascending=False).head(6).to_dict()
    s["share_of_congestion_in_top6_hours_pct"] = round(float(d[d.hour.isin(hr.sort_values(ascending=False).head(6).index)]["c"].sum() / d["c"].sum() * 100), 1)
    figs["by_site"], rate = fig_by_site(df, out_dir)
    s["site_rate_pct_min_max"] = [round(float(rate.c.min()), 2), round(float(rate.c.max()), 2)]
    figs["correlation"], corr = fig_correlation(df, out_dir)
    s["spearman_with_score"] = corr["congestion_score"].drop("congestion_score").round(2).to_dict()
    s["latency_spearman_with_score"] = {
        "pooled": round(float(df.latency_ms.corr(df.congestion_score, method="spearman")), 2),
        "within_region": {r: round(float(g.latency_ms.corr(g.congestion_score, method="spearman")), 2)
                          for r, g in df.groupby("region")},
        "median_latency_ms_by_region": df.groupby("region")["latency_ms"].median().round(1).to_dict()}
    strength = indicator_strength(df)
    s["indicator_auc"] = strength.round(3).to_dict(orient="records")
    figs["peak"], peak = fig_peak_behavior(df, out_dir); s["suggested_peak_hours"] = peak
    figs["dists"] = fig_feature_dists(df, out_dir)
    figs["temporal"], tm, lens = fig_temporal_structure(df, out_dir, steps)
    s["transition_matrix"] = tm.round(3).values.tolist()
    s["episodes"] = ({"count": 0} if len(lens) == 0 else
                     {"count": int(len(lens)), "median_min": float(np.median(lens)),
                      "share_le_10min_pct": round(float((lens <= 10).mean() * 100), 1),
                      "p90_min": float(np.percentile(lens, 90)), "max_min": float(lens.max())})
    s["persistence_accuracy"] = round(float((df.congestion_state_id == df.target_state_id).mean()), 4)
    cong_now = df.congestion_state_id >= 2
    s["p_congested_in_H_given_now"] = round(float(df.loc[cong_now, "target_binary"].mean()), 3)
    s["p_congested_in_H_given_not_now"] = round(float(df.loc[~cong_now, "target_binary"].mean()), 4)
    s["new_congestion_share_of_future_events_pct"] = round(float(((~cong_now) & (df.target_binary == 1)).sum() / (df.target_binary == 1).sum() * 100), 1)
    figs["onset"], onset = fig_onset(df, out_dir); s["onset_trajectory"] = onset
    if {"latitude", "longitude"} <= set(df.columns):           # positions are optional metadata
        s["spatial"] = spatial_stats(df, cfg)
        figs["cell_map"] = fig_cell_map(df, cfg, out_dir, s["spatial"])
    s["figures"] = figs
    (out_dir / "eda_summary.json").write_text(json.dumps(s, indent=2, default=str))
    logger.info("EDA complete: %d figures in %s", len(figs), out_dir)
    return s
