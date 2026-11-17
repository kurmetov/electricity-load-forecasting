"""Download, chunked CSV -> Parquet conversion, resampling to hours and days."""

from __future__ import annotations

import json
import time
import tracemalloc
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from elf import config as cfg


def download_raw(force: bool = False) -> None:
    """Download the UCI zip (about 250 MB) and unpack LD2011_2014.txt (about 680 MB)."""
    cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if cfg.RAW_TXT.exists() and not force:
        print(f"already there: {cfg.RAW_TXT}")
        return
    request = urllib.request.Request(cfg.RAW_URL, headers={"User-Agent": cfg.USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as resp, open(cfg.RAW_ZIP, "wb") as out:
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if done % (50 << 20) < (1 << 20):
                    print(f"  {done >> 20} MB")
    except urllib.error.URLError as exc:
        cfg.RAW_ZIP.unlink(missing_ok=True)
        raise SystemExit(f"download failed: {exc}. Download by hand from {cfg.RAW_URL} into {cfg.DATA_RAW}")
    with zipfile.ZipFile(cfg.RAW_ZIP) as zf:
        zf.extract("LD2011_2014.txt", cfg.DATA_RAW)
    print(f"saved {cfg.RAW_TXT} ({cfg.RAW_TXT.stat().st_size >> 20} MB)")


def read_raw_chunks(chunksize: int = 20_000, dtype: str = "float32"):
    """Yield chunks of the raw file: index = timestamp of the 15-minute reading, 370 columns in kW."""
    reader = pd.read_csv(cfg.RAW_TXT, sep=";", decimal=",", index_col=0, chunksize=chunksize)
    for chunk in reader:
        chunk.index = pd.to_datetime(chunk.index)
        chunk.index.name = "ts"
        yield chunk.astype(dtype)


def convert_to_parquet(chunksize: int = 20_000) -> dict:
    """Raw txt -> wide Parquet (float32) chunk by chunk; returns sizes, time and memory figures."""
    cfg.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tracemalloc.start()
    writer, n_rows, mem64, mem32 = None, 0, 0, 0
    for i, chunk in enumerate(read_raw_chunks(chunksize, dtype="float64")):
        if i == 0:
            mem64 = chunk.memory_usage(deep=True).sum()
        chunk = chunk.astype("float32")
        if i == 0:
            mem32 = chunk.memory_usage(deep=True).sum()
        table = pa.Table.from_pandas(chunk, preserve_index=True)
        if writer is None:
            writer = pq.ParquetWriter(cfg.LOAD_15MIN, table.schema, compression="zstd")
        writer.write_table(table)
        n_rows += len(chunk)
    writer.close()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n_cols = pq.read_schema(cfg.LOAD_15MIN).names.__len__() - 1
    stats = {
        "rows": n_rows,
        "clients": n_cols,
        "values": n_rows * n_cols,
        "chunksize": chunksize,
        "seconds": round(time.perf_counter() - t0, 1),
        "peak_mb": round(peak / 2**20, 1),
        "chunk_mb_float64": round(mem64 / 2**20, 2),
        "chunk_mb_float32": round(mem32 / 2**20, 2),
        "full_mb_float64_est": round(n_rows * n_cols * 8 / 2**20, 1),
        "full_mb_float32_est": round(n_rows * n_cols * 4 / 2**20, 1),
        "csv_mb": round(cfg.RAW_TXT.stat().st_size / 2**20, 1),
        "parquet_mb": round(cfg.LOAD_15MIN.stat().st_size / 2**20, 1),
    }
    return stats


def fix_dst(hourly: pd.DataFrame) -> pd.DataFrame:
    """Dataset note: on the March DST day hour 01 is zero for all clients, on the October day
    hour 01 holds two hours. Interpolate the first, halve the second. Only where the total shows
    the artefact (2011 does not have it)."""
    hourly = hourly.copy()
    hour = pd.Timedelta(hours=1)
    total = hourly.sum(axis=1)
    for day in dst_days(sorted(set(hourly.index.year))):
        ts = day + hour
        if ts - hour not in hourly.index or ts + hour not in hourly.index:
            continue
        neighbours = (total[ts - hour] + total[ts + hour]) / 2
        if day.month == 3 and total[ts] < 0.2 * neighbours:
            hourly.loc[ts] = (hourly.loc[ts - hour] + hourly.loc[ts + hour]) / 2
        elif day.month == 10 and total[ts] > 1.5 * neighbours:
            hourly.loc[ts] = hourly.loc[ts] / 2
    return hourly


def dst_days(years=range(2011, 2015)) -> list[pd.Timestamp]:
    """Last Sundays of March and October."""
    out = []
    for year in years:
        for month in (3, 10):
            day = pd.Timestamp(year, month, 31)
            out.append(day - pd.Timedelta(days=(day.dayofweek + 1) % 7))
    return out


def resample_load() -> dict:
    """15-min kW -> hourly and daily kWh. A 15-min reading in kW is 0.25 kWh per kW.
    A label marks the start of its interval: the March DST zeros sit exactly on 01:00-01:45.
    The file runs from 2011-01-01 00:15 to 2015-01-01 00:00, so the last bin (one reading) is dropped."""
    t0 = time.perf_counter()
    q = pd.read_parquet(cfg.LOAD_15MIN)
    hourly = (q.resample("h").sum() * 0.25).astype("float32").loc[: f"{cfg.END} 23:00"]
    del q
    hourly.to_parquet(cfg.LOAD_HOURLY, compression="zstd")
    daily = (hourly.resample("D").sum()).astype("float32")
    daily.index.name = "date"
    cfg.DATA_PROC.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cfg.LOAD_DAILY, compression="zstd")
    clients = client_table(daily)
    clients.to_csv(cfg.CLIENTS, index=False)
    return {
        "hourly_rows": len(hourly),
        "daily_rows": len(daily),
        "hourly_mb": round(cfg.LOAD_HOURLY.stat().st_size / 2**20, 1),
        "daily_mb": round(cfg.LOAD_DAILY.stat().st_size / 2**20, 2),
        "active_clients": int(clients["active"].sum()),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def client_table(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per client: first/last day with consumption, share of zero days, mean kWh/day.
    `active` = consumes from TRAIN_START (or earlier) to the end, under 5 % zero days since then."""
    rows = []
    for client in daily.columns:
        s = daily[client]
        nonzero = s[s > 0]
        first = nonzero.index.min() if len(nonzero) else pd.NaT
        last = nonzero.index.max() if len(nonzero) else pd.NaT
        since_train = s[cfg.TRAIN_START:]
        rows.append({
            "client_id": client,
            "first_day": first,
            "last_day": last,
            "zero_days_share": round(float((s == 0).mean()), 4),
            "zero_days_share_since_2012": round(float((since_train == 0).mean()), 4),
            "mean_kwh_day": round(float(since_train.mean()), 2),
        })
    clients = pd.DataFrame(rows)
    clients["active"] = (
        (clients["first_day"] <= pd.Timestamp(cfg.TRAIN_START))
        & (clients["last_day"] >= pd.Timestamp(cfg.END))
        & (clients["zero_days_share_since_2012"] < 0.05)
    )
    return clients


def load_hourly(columns: list[str] | None = None, clean: bool = True) -> pd.DataFrame:
    hourly = pd.read_parquet(cfg.LOAD_HOURLY, columns=columns)
    return fix_dst(hourly) if clean else hourly


def load_daily() -> pd.DataFrame:
    return pd.read_parquet(cfg.LOAD_DAILY)


def load_clients() -> pd.DataFrame:
    return pd.read_csv(cfg.CLIENTS, parse_dates=["first_day", "last_day"])


def save_json(obj: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def ingest(chunksize: int = 20_000) -> dict:
    """Download (if needed), convert to Parquet, resample. Logged as a runboard run."""
    from elf.tracking import Run

    download_raw()
    with Run("ingest", {"chunksize": chunksize, "dtype": "float32", "compression": "zstd"}, tags={"step": "S1"}) as run:
        stats = {"convert": convert_to_parquet(chunksize), "resample": resample_load()}
        flat = {f"{k}.{kk}": vv for k, v in stats.items() for kk, vv in v.items()}
        save_json(stats, cfg.RESULTS / "ingest.json")
        run.finish(flat)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ingest()
