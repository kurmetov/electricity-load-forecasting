"""Day-ahead hourly forecasts on a chronological hold-out (2014-10-01 .. 2014-12-31).

Every day of the test period is forecast at midnight for the next 24 hours, using data up to
the previous hour only. Baselines: naive (last value), seasonal naive 24 h and 168 h.
Models: ETS on 24 daily series, one per hour of the day (refit every day on a trailing window),
ridge regression on lags, gradient boosting on lags (sklearn HistGradientBoostingRegressor).
"""

from __future__ import annotations

import time
import tracemalloc
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from elf import config as cfg
from elf import data, external
from elf.labels import MODEL_NAMES

OUT = cfg.RESULTS / "forecast"
MODELS = ["naive", "snaive24", "snaive168", "ets", "ridge", "gbm"]
LAGS = [24, 48, 72, 168, 336]
CATEGORICAL = ["hour", "dow", "month"]
DEFAULTS = {"ridge_alpha": 1.0, "gbm_lr": 0.05, "gbm_leaves": 31, "gbm_iter": 300,
            "ets_window_days": 112, "ets_seasonal": "add"}
VALID_DAYS = ("2014-09-01", "2014-09-30")  # last month of the training part, used to choose the ETS setup


# ---------- metrics ----------

def metrics(y: np.ndarray, p: np.ndarray, scale: float) -> dict:
    """MAE, RMSE, MAPE (hours with y > 0 only), sMAPE, MASE (scale = in-sample MAE of seasonal naive 24 h)."""
    err = y - p
    nz = y > 0
    denom = np.abs(y) + np.abs(p)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(100 * np.mean(np.abs(err[nz]) / y[nz])) if nz.any() else np.nan,
        "smape": float(100 * np.mean(np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0))),
        "mase": float(np.mean(np.abs(err)) / scale) if scale > 0 else np.nan,
    }


def mase_scale(y_train: pd.Series, m: int = 24) -> float:
    return float(np.mean(np.abs(y_train.to_numpy()[m:] - y_train.to_numpy()[:-m])))


# ---------- features ----------

def hourly_calendar() -> pd.DataFrame:
    """Hourly exogenous columns: holiday flag and temperature."""
    cal = external.load_calendar()
    weather = pd.read_parquet(cfg.WEATHER)
    hol = cal["is_holiday"].reindex(weather.index.normalize()).to_numpy()
    return pd.DataFrame({"holiday": hol.astype("int8"), "temp": weather["temp"].to_numpy()}, index=weather.index)


def make_features(y: pd.Series, exog: pd.DataFrame) -> pd.DataFrame:
    """Only lags >= 24 h, so every hour of the next day is known at midnight.
    Temperature is the observed one (stands in for a weather forecast)."""
    X = pd.DataFrame(index=y.index)
    for lag in LAGS:
        X[f"lag{lag}"] = y.shift(lag)
    shifted = y.shift(24)
    X["mean24"] = shifted.rolling(24).mean()
    X["mean168"] = shifted.rolling(168).mean()
    X["hour"] = y.index.hour
    X["dow"] = y.index.dayofweek
    X["month"] = y.index.month
    X = X.join(exog)
    X["temp2"] = (X["temp"] - 18) ** 2  # heating and cooling both raise the load
    return X


def ridge_model(alpha: float):
    numeric = [*(f"lag{l}" for l in LAGS), "mean24", "mean168", "holiday", "temp", "temp2"]
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                              ("num", StandardScaler(), numeric)])
    return make_pipeline(prep, Ridge(alpha=alpha))


def gbm_model(lr: float, leaves: int, n_iter: int, columns: list[str]):
    cat_mask = [c in CATEGORICAL for c in columns]
    return HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, max_iter=n_iter,
                                         categorical_features=cat_mask, early_stopping=False, random_state=0)


# ---------- forecasters: each returns predictions for the test index ----------

def predict_baseline(y: pd.Series, kind: str) -> pd.Series:
    if kind == "naive":  # value of hour 23 of the previous day, held for the whole day
        last = y.shift(1).where(y.index.hour == 0)
        return last.ffill()
    return y.shift(24 if kind == "snaive24" else 168)


def predict_ets(y: pd.Series, test_days: pd.DatetimeIndex, window_days: int, seasonal: str | None) -> pd.Series:
    """ETS per hour of the day: the hourly series is cut into 24 daily series (00:00, 01:00, ...),
    each gets simple exponential smoothing with an additive weekly season (7 days) or none,
    fitted at every origin on the last `window_days` days and forecast one day ahead.
    One ETS with a 24 h season on the hourly series lost to seasonal naive on the validation month:
    its level follows the last evening hours and carries them into the whole next day."""
    wide = y.groupby([y.index.normalize(), y.index.hour]).first().unstack()  # day x hour
    out = []
    for day in test_days:
        hist = wide.loc[day - pd.Timedelta(days=window_days): day - pd.Timedelta(days=1)]
        values = []
        for hour in range(24):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(hist[hour].to_numpy(), trend=None, seasonal=seasonal,
                                           seasonal_periods=7 if seasonal else None,
                                           initialization_method="estimated").fit()
            values.append(fit.forecast(1)[0])
        out.append(pd.Series(values, index=pd.date_range(day, periods=24, freq="h")))
    return pd.concat(out)


def tune_ets(y: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Choose window and season of ETS by MAE on the validation month (inside the training part)."""
    days = pd.date_range(*VALID_DAYS, freq="D")
    truth = y.loc[VALID_DAYS[0]: f"{VALID_DAYS[1]} 23:00"].to_numpy()
    rows = []
    for window in (28, 56, 112):
        for seasonal in (None, "add"):
            pred = predict_ets(y.astype("float64"), days, window, seasonal).to_numpy()
            rows.append({"model": "ets", "ets_window_days": window, "ets_seasonal": seasonal or "none",
                         "valid_mae": float(np.mean(np.abs(truth - pred)))})
    snaive = predict_baseline(y, "snaive24").loc[VALID_DAYS[0]: f"{VALID_DAYS[1]} 23:00"].to_numpy()
    rows.append({"model": "snaive24", "valid_mae": float(np.mean(np.abs(truth - snaive)))})
    table = pd.DataFrame(rows)
    best = table[table["model"] == "ets"].sort_values("valid_mae").iloc[0]
    return {"ets_window_days": int(best["ets_window_days"]),
            "ets_seasonal": None if best["ets_seasonal"] == "none" else best["ets_seasonal"]}, table


def fit_predict(model: str, y: pd.Series, X: pd.DataFrame, params: dict) -> tuple[pd.Series, float]:
    """Returns test predictions and seconds spent fitting + predicting."""
    test = y.loc[cfg.TEST_START:].index
    t0 = time.perf_counter()
    if model in ("naive", "snaive24", "snaive168"):
        pred = predict_baseline(y, model).loc[test]
    elif model == "ets":
        days = pd.date_range(cfg.TEST_START, cfg.END, freq="D")
        pred = predict_ets(y, days, params["ets_window_days"], params["ets_seasonal"])
    else:
        train = X.loc[cfg.TRAIN_START: test[0] - pd.Timedelta(hours=1)].dropna()
        if model == "ridge":
            est = ridge_model(params["ridge_alpha"])
        else:
            est = gbm_model(params["gbm_lr"], params["gbm_leaves"], params["gbm_iter"], list(X.columns))
        est.fit(train, y.loc[train.index])
        pred = pd.Series(est.predict(X.loc[test]), index=test)
    pred = pred.clip(lower=0)  # consumption is never negative
    return pred, time.perf_counter() - t0


def evaluate_series(name: str, y: pd.Series, exog: pd.DataFrame, params: dict, models=MODELS) -> tuple[list[dict], pd.DataFrame]:
    y = y.astype("float64")
    X = make_features(y, exog)
    scale = mase_scale(y.loc[cfg.TRAIN_START:cfg.TEST_START].iloc[:-1])
    test_y = y.loc[cfg.TEST_START:]
    rows, preds = [], {"y_true": test_y}
    for model in models:
        pred, seconds = fit_predict(model, y, X, params)
        preds[model] = pred
        rows.append({"series": name, "model": model, **metrics(test_y.to_numpy(), pred.to_numpy(), scale),
                     "fit_s": seconds})
    frame = pd.DataFrame(preds).astype("float32")
    frame.index.name = "ts"
    return rows, frame.reset_index().assign(series=name)


def profile_memory(y: pd.Series, exog: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Peak Python-tracked memory (tracemalloc) and time of each model on one series."""
    y = y.astype("float64")
    X = make_features(y, exog)
    rows = []
    for model in MODELS:
        tracemalloc.start()
        _, seconds = fit_predict(model, y, X, params)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({"model": model, "peak_mb": round(peak / 2**20, 2), "seconds_with_tracemalloc": round(seconds, 3)})
    return pd.DataFrame(rows)


def tune(y: pd.Series, exog: pd.DataFrame, n_splits: int = 4) -> tuple[dict, pd.DataFrame]:
    """TimeSeriesSplit (expanding window, no shuffling) on the training part of the total series:
    ridge alpha and the GBM learning rate / leaves, chosen by mean MAE over the folds."""
    y = y.astype("float64")
    X = make_features(y, exog).loc[cfg.TRAIN_START: pd.Timestamp(cfg.TEST_START) - pd.Timedelta(hours=1)].dropna()
    target = y.loc[X.index]
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=24 * 28)
    candidates = [("ridge", {"ridge_alpha": a}) for a in (0.1, 1.0, 10.0, 100.0)]
    candidates += [("gbm", {"gbm_lr": lr, "gbm_leaves": lv}) for lr in (0.05, 0.1) for lv in (15, 31, 63)]
    rows = []
    for model, p in candidates:
        params = {**DEFAULTS, **p}
        maes = []
        for tr, va in splitter.split(X):
            est = ridge_model(params["ridge_alpha"]) if model == "ridge" else \
                gbm_model(params["gbm_lr"], params["gbm_leaves"], params["gbm_iter"], list(X.columns))
            est.fit(X.iloc[tr], target.iloc[tr])
            maes.append(np.mean(np.abs(target.iloc[va].to_numpy() - est.predict(X.iloc[va]))))
        rows.append({"model": model, **p, "cv_mae": float(np.mean(maes)), "cv_mae_std": float(np.std(maes))})
    table = pd.DataFrame(rows)
    best = dict(DEFAULTS)
    for model in ("ridge", "gbm"):
        top = table[table["model"] == model].sort_values("cv_mae").iloc[0]
        keys = ["ridge_alpha"] if model == "ridge" else ["gbm_lr", "gbm_leaves"]
        best.update({k: (int(top[k]) if k == "gbm_leaves" else float(top[k])) for k in keys})
    return best, table


# ---------- error analysis and figures ----------

def error_breakdown(preds: pd.DataFrame, clusters: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """sMAPE by month, day type and hour (all series), and MASE by cluster."""
    cal = external.load_calendar()
    long = preds.melt(id_vars=["ts", "series", "y_true"], value_vars=MODELS, var_name="model", value_name="y_pred")
    long["ae"] = (long["y_true"] - long["y_pred"]).abs()
    denom = long["y_true"].abs() + long["y_pred"].abs()
    long["sape"] = np.where(denom > 0, 200 * long["ae"] / denom.where(denom > 0, 1), 0)
    day = long["ts"].dt.normalize()
    is_hol = day.map(cal["is_holiday"]).fillna(False).astype(bool)
    long["day_type"] = np.select([is_hol, long["ts"].dt.dayofweek >= 5], ["holiday", "weekend"], "working day")
    long["month"] = long["ts"].dt.strftime("%Y-%m")
    long["hour"] = long["ts"].dt.hour
    out = {
        "by_month": long.pivot_table(index="month", columns="model", values="sape", aggfunc="mean")[MODELS],
        "by_day_type": long.pivot_table(index="day_type", columns="model", values="sape", aggfunc="mean")[MODELS],
        "by_hour": long.pivot_table(index="hour", columns="model", values="sape", aggfunc="mean")[MODELS],
    }
    # worst days for the best model: when do forecasts fail
    daily_err = long[long["model"] == "gbm"].groupby(day)["sape"].mean().sort_values(ascending=False)
    out["worst_days_gbm"] = daily_err.head(10).rename("smape").to_frame().join(cal[["holiday_name"]])
    if clusters is not None:
        m = pd.read_csv(OUT / "metrics.csv")
        m = m.merge(clusters, left_on="series", right_on="client_id")
        out["mase_by_cluster"] = m.pivot_table(index="cluster", columns="model", values="mase", aggfunc="median")[MODELS]
    return {k: v.round(3) for k, v in out.items()}


def make_figures(preds: pd.DataFrame, summary: pd.DataFrame, breakdown: dict) -> None:
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    total = preds[preds["series"] == "total"].set_index("ts")
    week = total.loc["2014-12-15":"2014-12-28"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(week.index, week["y_true"] / 1000, color="black", lw=1.5, label="actual")
    for model in ("snaive168", "ets", "ridge", "gbm"):
        ax.plot(week.index, week[model] / 1000, lw=1, label=MODEL_NAMES[model])
    ax.set_ylabel("MWh per hour")
    ax.set_title("Day-ahead forecast, total load, 15–28 December 2014")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "07_forecast_total.png", dpi=150)
    plt.close(fig)

    pfig = go.Figure()
    pfig.add_scatter(x=total.index, y=total["y_true"], name="actual", line={"color": "black"})
    for model in MODELS:
        pfig.add_scatter(x=total.index, y=total[model], name=MODEL_NAMES[model], visible="legendonly" if model in ("naive", "snaive24") else True)
    pfig.update_layout(title="Total load: actual and forecasts, kWh per hour", xaxis_rangeslider_visible=True)
    pfig.write_html(cfg.FIGURES / "07_forecast_total.html", include_plotlyjs="cdn")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    breakdown["by_month"].T.rename(index=MODEL_NAMES).plot.bar(ax=axes[0], rot=25)
    axes[0].set_title("sMAPE by month, %")
    axes[0].set_xlabel("")
    plt.setp(axes[0].get_xticklabels(), ha="right")
    axes[0].legend(fontsize=7)
    breakdown["by_hour"][["snaive168", "ets", "ridge", "gbm"]].rename(columns=MODEL_NAMES).plot(ax=axes[1])
    axes[1].set_title("sMAPE by hour of day, %")
    axes[1].set_xlabel("hour")
    lo, hi = axes[1].get_ylim()
    axes[1].set_ylim(lo, hi + 0.35 * (hi - lo))  # free band on top so the legend does not cover the lines
    axes[1].legend(fontsize=7, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "08_errors.png", dpi=150)
    plt.close(fig)


def run_forecast(max_clients: int | None = None, n_jobs: int = -1, models=MODELS) -> pd.DataFrame:
    from elf.tracking import Run

    OUT.mkdir(parents=True, exist_ok=True)
    clients = data.load_clients()
    active = clients.loc[clients["active"], "client_id"].tolist()[:max_clients]
    hourly = data.load_hourly(columns=active)
    exog = hourly_calendar()
    series = {"total": hourly.sum(axis=1), **{c: hourly[c] for c in active}}
    del hourly

    tuned, cv_table = tune(series["total"], exog)
    cv_table.to_csv(OUT / "tuning_cv.csv", index=False)
    ets_params, ets_table = tune_ets(series["total"])
    ets_table.to_csv(OUT / "tuning_ets.csv", index=False)
    tuned.update(ets_params)
    config = {"train": [cfg.TRAIN_START, cfg.TEST_START], "test": [cfg.TEST_START, cfg.END], "horizon_h": 24,
              "origin": "every midnight of the test period", "lags": LAGS, "params": tuned,
              "tuning": "ridge, gbm: TimeSeriesSplit(4, test_size=28 days) on total series, mean MAE; "
                        f"ets: MAE on {VALID_DAYS[0]}..{VALID_DAYS[1]} of total series",
              "series": len(series), "models": list(models), "mase_scale": "in-sample MAE of seasonal naive 24 h"}
    with Run("forecast", config, tags={"step": "S4"}, dims=["model", "series"]) as run:
        for _, r in cv_table.iterrows():
            run.log({"cv_mae": r["cv_mae"]}, model=r["model"], series="total_cv", param=str(r.drop(["model", "cv_mae", "cv_mae_std"]).dropna().to_dict()))
        for _, r in ets_table.iterrows():
            run.log({"valid_mae": r["valid_mae"]}, model=r["model"], series="total_valid", param=str(r.drop(["model", "valid_mae"]).dropna().to_dict()))
        t0 = time.perf_counter()
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(evaluate_series)(name, y, exog, tuned, models) for name, y in series.items())
        wall = time.perf_counter() - t0
        rows = [r for rs, _ in results for r in rs]
        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(OUT / "metrics.csv", index=False)
        preds = pd.concat([p for _, p in results], ignore_index=True)
        preds["series"] = preds["series"].astype("category")
        preds.to_parquet(OUT / "forecasts.parquet", compression="zstd")
        for r in rows:
            run.log({k: r[k] for k in ("mae", "rmse", "mape", "smape", "mase", "fit_s")}, model=r["model"], series=r["series"])

        clients_only = metrics_df[metrics_df["series"] != "total"]
        summary = clients_only.groupby("model").agg(
            mase_mean=("mase", "mean"), mase_median=("mase", "median"), smape_mean=("smape", "mean"),
            mape_median=("mape", "median"), fit_s_mean=("fit_s", "mean")).reindex(list(models))
        total = metrics_df[metrics_df["series"] == "total"].set_index("model")[["mae", "rmse", "mape", "smape", "mase", "fit_s"]]
        summary = total.add_prefix("total_").join(summary).round(4)
        summary["wins_mase"] = clients_only.loc[clients_only.groupby("series")["mase"].idxmin(), "model"].value_counts().reindex(summary.index).fillna(0).astype(int)
        summary.to_csv(OUT / "summary.csv")
        data.save_json({**tuned, "n_series": len(series), "n_clients": len(series) - 1}, OUT / "params.json")
        mem = profile_memory(series["total"], exog, tuned)
        mem.to_csv(OUT / "memory_profile.csv", index=False)

        clusters_path = cfg.RESULTS / "clusters" / "clusters.csv"
        clusters = pd.read_csv(clusters_path) if clusters_path.exists() else None
        breakdown = error_breakdown(preds, clusters)
        for key, table in breakdown.items():
            table.to_csv(OUT / f"errors_{key}.csv")
        make_figures(preds, summary, breakdown)

        flat = {"wall_s": round(wall, 1), "series": len(series)}
        for model, r in summary.iterrows():
            flat.update({f"{model}.{k}": v for k, v in r.items()})
        for _, r in mem.iterrows():
            flat[f"{r['model']}.peak_mb"] = r["peak_mb"]
        flat.update({f"param.{k}": v for k, v in tuned.items()})
        run.finish(flat)
    print(summary)
    return summary


if __name__ == "__main__":
    import sys

    run_forecast(max_clients=int(sys.argv[1]) if len(sys.argv) > 1 else None)
