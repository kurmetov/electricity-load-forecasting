"""Streamlit dashboard: `streamlit run app.py`. Reads only saved files (data/processed, results/)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from elf import config as cfg
from elf.forecast import MODELS, metrics
from elf.labels import pretty

st.set_page_config(page_title="Electricity Load Forecasting", layout="wide")


@st.cache_data
def load() -> dict:
    clusters = pd.read_csv(cfg.RESULTS / "clusters" / "clusters.csv")
    return {
        "daily": pd.read_parquet(cfg.LOAD_DAILY),
        "clients": pd.read_csv(cfg.CLIENTS).merge(clusters, on="client_id", how="left"),
        "centres": pd.read_parquet(cfg.RESULTS / "clusters" / "centres.parquet"),
        "cluster_info": pd.read_csv(cfg.RESULTS / "clusters" / "summary.csv"),
        "forecasts": pd.read_parquet(cfg.RESULTS / "forecast" / "forecasts.parquet"),
        "metrics": pd.read_csv(cfg.RESULTS / "forecast" / "metrics.csv"),
        "summary": pd.read_csv(cfg.RESULTS / "forecast" / "summary.csv"),
    }


try:
    d = load()
except FileNotFoundError as exc:
    st.error(f"Missing file {exc.filename}. Run the pipeline: `python main.py all` (see README).")
    st.stop()

clients = d["clients"]
active = clients[clients["active"]]
cluster_names = d["cluster_info"].set_index("cluster")["name"].to_dict()

# ---------- filters ----------
st.sidebar.header("Filters")
first, last = d["daily"].index.min().date(), d["daily"].index.max().date()
period = st.sidebar.date_input("Period", (pd.Timestamp(cfg.TRAIN_START).date(), last), min_value=first, max_value=last)
start, end = (pd.Timestamp(period[0]), pd.Timestamp(period[1])) if len(period) == 2 else (pd.Timestamp(first), pd.Timestamp(last))
mode = st.sidebar.radio("Object", ["All clients (total)", "Cluster", "Client"])
if mode == "Client":
    labels = {f"{r.client_id} (cluster {int(r.cluster)})": r.client_id for r in active.itertuples()}
    members = [labels[st.sidebar.selectbox("Client", list(labels))]]
    title = members[0]
elif mode == "Cluster":
    cluster = st.sidebar.selectbox("Cluster", list(cluster_names), format_func=lambda c: f"{c}: {cluster_names[c]}")
    members = active.loc[active["cluster"] == cluster, "client_id"].tolist()
    title = f"cluster {cluster} ({len(members)} clients)"
else:
    members = None
    title = "all clients"
series_kind = st.sidebar.radio("Series", ["Daily, kWh", "7-day moving average", "30-day moving average"])
models = st.sidebar.multiselect("Forecast models", MODELS, default=["snaive168", "ets", "ridge", "gbm"], format_func=pretty)

tab_load, tab_fc, tab_cl, tab_models = st.tabs(["Consumption", "Forecast", "Clusters", "Model comparison"])

# ---------- consumption ----------
with tab_load:
    daily = d["daily"].loc[start:end]
    y = daily.sum(axis=1) if members is None else daily[members].sum(axis=1)
    if series_kind != "Daily, kWh":
        y = y.rolling(7 if "7" in series_kind else 30).mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total for the period, MWh", f"{y.sum() / 1000:,.0f}" if series_kind.startswith("Daily") else "—")
    c2.metric("Mean per day, kWh", f"{y.mean():,.0f}")
    c3.metric("Maximum, kWh", f"{y.max():,.0f}")
    fig = px.line(x=y.index, y=y.to_numpy(), labels={"x": "date", "y": "kWh per day"}, title=f"{series_kind}: {title}")
    fig.update_xaxes(tickformat="%Y-%m-%d")
    st.plotly_chart(fig, width="stretch")

# ---------- forecast ----------
with tab_fc:
    fc = d["forecasts"]
    test_start, test_end = max(start, pd.Timestamp(cfg.TEST_START)), min(end + pd.Timedelta(hours=23), fc["ts"].max())
    if test_start > test_end:
        st.info(f"Forecasts exist only for the hold-out period {cfg.TEST_START} – {cfg.END}. Widen the period.")
    else:
        if members is None:
            part = fc[fc["series"] == "total"]
        else:
            part = fc[fc["series"].isin(members)]
        agg = part.groupby("ts")[["y_true", *MODELS]].sum().loc[test_start:test_end]
        fig = go.Figure()
        fig.add_scatter(x=agg.index, y=agg["y_true"], name="actual", line={"color": "black", "width": 2})
        for m in models:
            fig.add_scatter(x=agg.index, y=agg[m], name=pretty(m), line={"width": 1})
        fig.update_layout(title=f"Hourly day-ahead forecast: {title}", yaxis_title="kWh per hour",
                          xaxis_rangeslider_visible=True, xaxis_tickformat="%Y-%m-%d")
        st.plotly_chart(fig, width="stretch")
        # metrics of the shown (aggregated) series; MASE scale is not stored, so MAE of snaive24 stands in
        scale = (agg["y_true"] - agg["snaive24"]).abs().mean()
        table = pd.DataFrame({pretty(m): metrics(agg["y_true"].to_numpy(), agg[m].to_numpy(), scale) for m in MODELS}).T
        table = table.rename(columns=pretty).rename(columns={"MASE": "MAE / MAE of seasonal naive 24 h"})
        st.caption("Metrics on the chosen part of the hold-out period (for a cluster, on the sum of its clients).")
        st.dataframe(table.round(3), width="stretch")

# ---------- clusters ----------
with tab_cl:
    centres = d["centres"]
    fig = go.Figure()
    for c, row in centres.iterrows():
        fig.add_scatter(x=list(range(48)), y=row.to_numpy(), name=f"{c}: {cluster_names[c]}")
    fig.add_vline(x=23.5, line_dash="dot")
    fig.update_layout(title="Cluster centres: hours 0–23 weekdays, 24–47 weekends (share of client mean)",
                      xaxis_title="hour (weekdays | weekends)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(d["cluster_info"].rename(columns=pretty), width="stretch")
    if mode == "Cluster":
        st.write(f"Clients of cluster {cluster}:", ", ".join(members))

# ---------- model comparison ----------
with tab_models:
    st.dataframe(d["summary"].set_index("model").rename(index=pretty, columns=pretty).rename_axis("Model").round(3), width="stretch")
    m = d["metrics"]
    m = m[m["series"] != "total"] if members is None else m[m["series"].isin(members)]
    st.plotly_chart(px.box(m.assign(model=m["model"].map(pretty)), x="model", y="mase", points=False,
                           category_orders={"model": [pretty(x) for x in MODELS]}, labels={"model": "model", "mase": "MASE"},
                           title=f"MASE by client ({title})"), width="stretch")
