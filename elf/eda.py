"""EDA: index checks, descriptive statistics, zeros and outliers, STL, ADF/KPSS, ACF/PACF, figures."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from elf import config as cfg
from elf import data, external

OUT = cfg.RESULTS / "eda"


def index_checks() -> dict:
    """Frequency, duplicates, gaps and order of the raw 15-minute index."""
    ts = pq.read_table(cfg.LOAD_15MIN, columns=["ts"]).column("ts").to_pandas()
    idx = pd.DatetimeIndex(ts)
    steps = idx.to_series().diff().dropna().value_counts()
    expected = pd.date_range(idx.min(), idx.max(), freq="15min")
    return {
        "rows": len(idx),
        "start": str(idx.min()),
        "end": str(idx.max()),
        "inferred_freq": pd.infer_freq(idx[:1000]),
        "monotonic": bool(idx.is_monotonic_increasing),
        "duplicates": int(idx.duplicated().sum()),
        "missing_steps": int(len(expected.difference(idx))),
        "step_counts": {str(k): int(v) for k, v in steps.items()},
    }


def missing_and_zeros(clients: pd.DataFrame) -> dict:
    """NaN in the raw parquet and zero stretches (client not yet connected or gone)."""
    nan_total = 0
    pf = pq.ParquetFile(cfg.LOAD_15MIN)
    for i in range(pf.num_row_groups):
        nan_total += int(pf.read_row_group(i).to_pandas().isna().sum().sum())
    first_year = clients["first_day"].dt.year.value_counts().sort_index()
    return {
        "nan_values": nan_total,
        "clients_total": len(clients),
        "clients_active": int(clients["active"].sum()),
        "clients_by_first_year": {str(k): int(v) for k, v in first_year.items()},
        "clients_never_consume": int(clients["first_day"].isna().sum()),
        "clients_end_before_2014_12_31": int((clients["last_day"] < pd.Timestamp(cfg.END)).sum()),
        "mean_zero_days_share": round(float(clients["zero_days_share"].mean()), 4),
    }


def outliers(daily: pd.DataFrame, active: list[str], hourly_raw_total: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Robust z-score > 5 on daily kWh of active clients (since 2012) + the DST hours."""
    d = daily.loc[cfg.TRAIN_START:, active]
    med = d.median()
    mad = (d - med).abs().median() * 1.4826
    z = (d - med).abs() / mad.replace(0, np.nan)
    flags = z > 5
    per_client = flags.sum().sort_values(ascending=False)
    by_date = flags.sum(axis=1).sort_values(ascending=False)
    dst = []
    for day in data.dst_days():
        ts = day + pd.Timedelta(hours=1)
        dst.append({"ts": str(ts),
                    "total_kwh": round(float(hourly_raw_total.loc[ts]), 1),
                    "neighbours_mean_kwh": round(float(hourly_raw_total.loc[[ts - pd.Timedelta(hours=1), ts + pd.Timedelta(hours=1)]].mean()), 1)})
    summary = {
        "method": "robust z = |x - median| / (1.4826 * MAD) > 5, daily kWh, active clients, 2012-2014",
        "flagged_values": int(flags.values.sum()),
        "flagged_share_pct": round(100 * float(flags.values.mean()), 3),
        "clients_with_flags": int((per_client > 0).sum()),
        "top_dates": {str(k.date()): int(v) for k, v in by_date.head(10).items()},
        "dst_hours": dst,
    }
    table = pd.DataFrame({"flagged_days": per_client, "median_kwh": med.round(1), "mad_kwh": mad.round(1)})
    return summary, table


def stationarity(series: pd.Series, name: str) -> dict:
    """ADF (H0: unit root) and KPSS (H0: stationary) on a series."""
    s = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS warns when p is outside its lookup table
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    if adf_p < 0.05 and kpss_p > 0.05:
        verdict = "stationary"
    elif adf_p >= 0.05 and kpss_p <= 0.05:
        verdict = "non-stationary"
    else:
        verdict = "mixed"
    return {"series": name, "adf_stat": round(adf_stat, 3), "adf_p": round(adf_p, 4),
            "kpss_stat": round(kpss_stat, 3), "kpss_p": round(kpss_p, 4), "verdict": verdict}


def stl_strength(series: pd.Series, period: int) -> tuple[dict, object]:
    """STL and the strength of trend and seasonality (Hyndman & Athanasopoulos, ch. 4.3)."""
    res = STL(series, period=period, robust=True).fit()
    r, s, t = res.resid, res.seasonal, res.trend
    strength = {
        "period": period,
        "trend_strength": round(max(0.0, 1 - r.var() / (t + r).var()), 3),
        "seasonal_strength": round(max(0.0, 1 - r.var() / (s + r).var()), 3),
    }
    return strength, res


def make_figures(daily_total: pd.Series, hourly_total: pd.Series, stl_daily, cal: pd.DataFrame) -> list[str]:
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    saved = []
    mwh = daily_total / 1000
    roll7, roll30 = mwh.rolling(7).mean(), mwh.rolling(30).mean()

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(mwh.index, mwh, lw=0.5, color="0.6", label="daily")
    ax.plot(roll7.index, roll7, lw=1.2, label="7-day moving average")
    ax.plot(roll30.index, roll30, lw=1.8, label="30-day moving average")
    ax.set_ylabel("MWh per day")
    ax.set_title("Total daily consumption of 370 clients")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "01_daily_total.png", dpi=150)
    plt.close(fig)

    pfig = go.Figure()
    pfig.add_scatter(x=mwh.index, y=mwh, name="daily", line={"width": 1, "color": "#999"})
    pfig.add_scatter(x=roll7.index, y=roll7, name="7 days")
    pfig.add_scatter(x=roll30.index, y=roll30, name="30 days")
    pfig.update_layout(title="Total daily consumption, MWh", xaxis_rangeslider_visible=True)
    pfig.write_html(cfg.FIGURES / "01_daily_total.html", include_plotlyjs="cdn")
    saved += ["01_daily_total.png", "01_daily_total.html"]

    fig = stl_daily.plot()
    fig.set_size_inches(10, 6)
    fig.suptitle("STL of the daily series (period 7), MWh per day", y=1.0)
    for ax in fig.axes:  # kWh -> MWh on the ticks, no 1e6 offset
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}"))
        ax.set_ylabel(f"{ax.get_ylabel() or 'Load'}, MWh")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "02_stl_daily.png", dpi=150)
    plt.close(fig)
    saved.append("02_stl_daily.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 5.4))
    plot_acf(hourly_total, lags=336, ax=axes[0, 0], markersize=2, title="ACF, hourly series (up to 336 h)")
    plot_pacf(hourly_total, lags=72, ax=axes[0, 1], markersize=2, method="ywm", title="PACF, hourly series (up to 72 h)")
    plot_acf(daily_total, lags=60, ax=axes[1, 0], markersize=3, title="ACF, daily series (up to 60 days)")
    plot_pacf(daily_total, lags=30, ax=axes[1, 1], markersize=3, method="ywm", title="PACF, daily series (up to 30 days)")
    for ax, unit in zip(axes.flat, ("h", "h", "days", "days")):
        ax.set_xlabel(f"lag, {unit}")
        ax.set_ylabel("correlation")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "03_acf_pacf.png", dpi=150)
    plt.close(fig)
    saved.append("03_acf_pacf.png")

    prof = hourly_total.groupby([hourly_total.index.dayofweek, hourly_total.index.hour]).mean().unstack() / 1000
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(prof.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2), range(0, 24, 2))
    ax.set_xlabel("hour")
    ax.set_title("Mean hourly consumption by day of week, MWh")
    fig.colorbar(im, ax=ax, label="MWh per hour")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "04_heatmap_hour_dow.png", dpi=150)
    plt.close(fig)
    saved.append("04_heatmap_hour_dow.png")

    df = cal.join(mwh.rename("load")).loc[cfg.TRAIN_START:]
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, mask, color in (("working day", ~df["is_weekend"] & ~df["is_holiday"], "C0"),
                               ("weekend", df["is_weekend"] & ~df["is_holiday"], "C1"),
                               ("holiday", df["is_holiday"], "C3")):
        ax.scatter(df.loc[mask, "temp_mean"], df.loc[mask, "load"], s=6, alpha=0.6, label=label, color=color)
    ax.set_xlabel("daily mean temperature, °C (Lisbon)")
    ax.set_ylabel("MWh per day")
    ax.set_title("Consumption and temperature, 2012–2014")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "05_temp_vs_load.png", dpi=150)
    plt.close(fig)
    saved.append("05_temp_vs_load.png")
    return saved


def run_eda() -> dict:
    from elf.tracking import Run

    OUT.mkdir(parents=True, exist_ok=True)
    with Run("eda", {"outlier_rule": "robust z > 5", "stl_periods": [7, 24], "since": cfg.TRAIN_START}, tags={"step": "S2"}) as run:
        clients = data.load_clients()
        active = clients.loc[clients["active"], "client_id"].tolist()
        daily = data.load_daily()
        hourly_raw = data.load_hourly(clean=False)
        hourly_raw_total = hourly_raw.sum(axis=1)
        hourly = data.fix_dst(hourly_raw)
        del hourly_raw
        hourly_total = hourly.sum(axis=1).loc[cfg.TRAIN_START:]
        daily_total = daily.sum(axis=1).loc[cfg.TRAIN_START:]

        # descriptive statistics: per client (hourly kWh since 2012) and for the total
        per_client = hourly.loc[cfg.TRAIN_START:].describe().T.round(2)
        per_client.index.name = "client_id"
        per_client.to_csv(OUT / "client_stats.csv")
        totals = pd.DataFrame({"hourly_total_kwh": hourly_total.describe(), "daily_total_kwh": daily_total.describe()}).round(1)
        totals.to_csv(OUT / "total_stats.csv")
        del hourly

        weather = external.fetch_weather()
        cal = external.build_calendar(weather)
        cal.to_csv(OUT / "calendar.csv")

        stl_d, stl_daily = stl_strength(daily_total, 7)
        stl_h, _ = stl_strength(hourly_total.loc["2014-01-01":], 24)
        tests = [
            stationarity(daily_total, "daily total"),
            stationarity(daily_total.diff(), "daily total, diff 1"),
            stationarity(daily_total.diff(7), "daily total, diff 7"),
            stationarity(hourly_total.loc["2014-01-01":], "hourly total 2014"),
            stationarity(hourly_total.loc["2014-01-01":].diff(24), "hourly total 2014, diff 24"),
        ]
        pd.DataFrame(tests).to_csv(OUT / "stationarity.csv", index=False)
        out_summary, out_table = outliers(daily, active, hourly_raw_total)
        out_table.to_csv(OUT / "outliers_by_client.csv")

        result = {
            "index": index_checks(),
            "missing_zeros": missing_and_zeros(clients),
            "outliers": out_summary,
            "stl_daily": stl_d,
            "stl_hourly_2014": stl_h,
            "stationarity": tests,
            "external": external.external_correlations(daily_total, cal),
            "figures": make_figures(daily_total, hourly_total, stl_daily, cal),
        }
        data.save_json(result, OUT / "eda.json")
        flat = {}
        for key in ("index", "missing_zeros", "outliers", "stl_daily", "stl_hourly_2014", "external"):
            flat.update({f"{key}.{k}": v for k, v in result[key].items() if isinstance(v, (int, float, str, bool))})
        for t in tests:
            run.log({"adf_p": t["adf_p"], "kpss_p": t["kpss_p"]}, series=t["series"])
        run.finish(flat)
    return result


if __name__ == "__main__":
    run_eda()
