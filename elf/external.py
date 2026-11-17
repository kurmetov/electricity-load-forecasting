"""External factors: hourly temperature in Lisbon (Open-Meteo archive) and Portuguese holidays."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import holidays
import numpy as np
import pandas as pd

from elf import config as cfg

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather() -> pd.DataFrame:
    """Hourly 2 m temperature for Lisbon, local time. No API key; neutral User-Agent."""
    if cfg.WEATHER.exists():
        return pd.read_parquet(cfg.WEATHER)
    params = {
        **cfg.LISBON,
        "start_date": cfg.START,
        "end_date": cfg.END,
        "hourly": "temperature_2m",
        "timezone": "Europe/Lisbon",
    }
    url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": cfg.USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Open-Meteo request failed: {exc}")
    weather = pd.DataFrame({"ts": pd.to_datetime(payload["hourly"]["time"]),
                            "temp": payload["hourly"]["temperature_2m"]})
    # local time: the October DST hour comes twice, the March one is missing -> same grid as the load data
    weather = weather.drop_duplicates("ts").set_index("ts")
    grid = pd.date_range(cfg.START, f"{cfg.END} 23:00", freq="h", name="ts")
    weather = weather.reindex(grid).interpolate(limit=3).astype("float32")
    cfg.DATA_PROC.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(cfg.WEATHER)
    return weather


def build_calendar(weather: pd.DataFrame) -> pd.DataFrame:
    """One row per day: weekday, weekend, Portuguese public holiday, daily temperature."""
    days = pd.date_range(cfg.START, cfg.END, freq="D", name="date")
    pt = holidays.Portugal(years=range(days.year.min(), days.year.max() + 1))
    temp = weather["temp"].resample("D").agg(["mean", "min", "max"]).add_prefix("temp_")
    cal = pd.DataFrame(index=days)
    cal["dow"] = days.dayofweek
    cal["is_weekend"] = cal["dow"] >= 5
    cal["holiday_name"] = [pt.get(d) for d in days.date]
    cal["is_holiday"] = cal["holiday_name"].notna()
    cal = cal.join(temp.round(2))
    cal.to_parquet(cfg.CALENDAR)
    return cal


def load_calendar() -> pd.DataFrame:
    return pd.read_parquet(cfg.CALENDAR)


def external_correlations(daily_total: pd.Series, cal: pd.DataFrame) -> dict:
    """Correlation of the daily total load with temperature, and the holiday effect on working days."""
    df = cal.join(daily_total.rename("load"))[cfg.TRAIN_START:]
    workdays = df[~df["is_weekend"]]
    # U-shape: heating below ~16 C, cooling above ~22 C -> correlate with distance from comfort band
    df["hdd"] = np.clip(16 - df["temp_mean"], 0, None)
    df["cdd"] = np.clip(df["temp_mean"] - 22, 0, None)
    out = {
        "pearson_temp": df["load"].corr(df["temp_mean"]),
        "spearman_temp": df["load"].corr(df["temp_mean"], method="spearman"),
        "pearson_hdd": df["load"].corr(df["hdd"]),
        "pearson_cdd": df["load"].corr(df["cdd"]),
        "workday_mean_mwh": workdays.loc[~workdays["is_holiday"], "load"].mean() / 1000,
        "holiday_mean_mwh": workdays.loc[workdays["is_holiday"], "load"].mean() / 1000,
        "weekend_mean_mwh": df.loc[df["is_weekend"], "load"].mean() / 1000,
        "n_holidays_on_workdays": int(workdays["is_holiday"].sum()),
    }
    out["holiday_drop_pct"] = 100 * (1 - out["holiday_mean_mwh"] / out["workday_mean_mwh"])
    return {k: round(float(v), 4) for k, v in out.items()}
