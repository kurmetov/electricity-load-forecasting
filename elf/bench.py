"""Comparisons by time and memory: reading CSV vs chunked CSV vs Parquet; pandas vs SQLite aggregation.

Memory = peak RSS of a fresh process while it runs one method minus its RSS before the call
(Linux: the peak is reset through /proc/self/clear_refs). tracemalloc does not see pyarrow's
allocator, so RSS is the fair measure across CSV and Parquet.
"""

from __future__ import annotations

import multiprocessing as mp
import statistics
import time

import pandas as pd

from elf import config as cfg
from elf import data
from elf.db import connect

OUT = cfg.RESULTS / "bench"
SQL_MONTHLY = """SELECT c.cluster_id, substr(l.date, 1, 7) AS month, SUM(l.kwh) / 1000 AS mwh
                 FROM load_daily l JOIN clients c ON c.client_id = l.client_id
                 WHERE l.date >= '2012-01-01' AND c.cluster_id IS NOT NULL
                 GROUP BY c.cluster_id, month"""


def csv_full():
    return pd.read_csv(cfg.RAW_TXT, sep=";", decimal=",", index_col=0, parse_dates=[0])


def csv_chunks_float32():
    return pd.concat(data.read_raw_chunks(20_000, "float32"))


def csv_chunks_to_daily():
    """Streaming: each chunk is reduced to daily sums at once, so only one chunk is in memory."""
    parts = [chunk.resample("D").sum() * 0.25 for chunk in data.read_raw_chunks(20_000, "float32")]
    return pd.concat(parts).groupby(level=0).sum()


def parquet_full():
    return pd.read_parquet(cfg.LOAD_15MIN)


def parquet_10_columns():
    return pd.read_parquet(cfg.LOAD_15MIN, columns=[f"MT_{i:03d}" for i in range(1, 11)])


def monthly_pandas():
    clusters = pd.read_csv(cfg.RESULTS / "clusters" / "clusters.csv")[["client_id", "cluster"]]
    daily = data.load_daily().loc[cfg.TRAIN_START:]
    long = daily.reset_index().melt(id_vars="date", var_name="client_id", value_name="kwh")
    long = long.merge(clusters, on="client_id")
    out = (long.groupby(["cluster", long["date"].dt.to_period("M")])["kwh"].sum() / 1000).rename("mwh").reset_index()
    return out


def monthly_sqlite():
    con = connect()
    out = pd.read_sql_query(SQL_MONTHLY, con)
    con.close()
    return out


METHODS = {
    "csv, pandas default (float64)": csv_full,
    "csv, chunks + float32": csv_chunks_float32,
    "csv, chunks -> daily sums (streaming)": csv_chunks_to_daily,
    "parquet zstd, float32": parquet_full,
    "parquet, 10 columns only": parquet_10_columns,
    "pandas: parquet -> melt -> merge -> groupby": monthly_pandas,
    "sqlite: JOIN + GROUP BY": monthly_sqlite,
}


def _rss_kb(field: str) -> int:
    for line in open("/proc/self/status"):
        if line.startswith(field):
            return int(line.split()[1])
    raise RuntimeError(f"{field} not found in /proc/self/status")


def _child(name: str) -> tuple[float, float, int]:
    with open("/proc/self/clear_refs", "w") as f:  # reset VmHWM to the current RSS
        f.write("5")
    before = _rss_kb("VmRSS:")
    result = METHODS[name]()
    peak = _rss_kb("VmHWM:")
    return (peak - before) / 1024, result.memory_usage(deep=True).sum() / 2**20, len(result)


def measure(name: str, repeat: int) -> dict:
    """Peak RSS growth and frame size in a fresh process, then median wall time of `repeat` runs here."""
    with mp.get_context("spawn").Pool(1) as pool:
        rss_mb, frame_mb, rows = pool.apply(_child, (name,))
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        METHODS[name]()
        times.append(time.perf_counter() - t0)
    return {"method": name, "seconds": round(statistics.median(times), 3), "peak_rss_mb": round(rss_mb, 1),
            "frame_mb": round(frame_mb, 1), "result_rows": rows}


def run_bench() -> dict:
    from elf.tracking import Run

    OUT.mkdir(parents=True, exist_ok=True)
    with Run("bench", {"repeat_csv": 2, "repeat_other": 5, "memory": "peak RSS growth in a spawned process"},
             tags={"step": "S6"}) as run:
        formats = pd.DataFrame([measure(n, 2 if n.startswith("csv") else 5) for n in list(METHODS)[:5]])
        formats.insert(1, "file_mb", [round(p.stat().st_size / 2**20, 1) for p in [cfg.RAW_TXT] * 3 + [cfg.LOAD_15MIN] * 2])
        formats.to_csv(OUT / "read_formats.csv", index=False)
        sql = pd.DataFrame([measure(n, 5) for n in list(METHODS)[5:]])
        a, b = monthly_pandas(), monthly_sqlite()
        sql["same_result"] = bool(abs(a["mwh"].sum() - b["mwh"].sum()) < 1e-3 * a["mwh"].sum() and len(a) == len(b))
        sql.to_csv(OUT / "pandas_vs_sql.csv", index=False)
        summary = {}
        for table in (formats, sql):
            for _, r in table.iterrows():
                run.log({"seconds": r["seconds"], "peak_rss_mb": r["peak_rss_mb"], "frame_mb": r["frame_mb"]}, method=r["method"])
                summary[f"{r['method']}.seconds"] = r["seconds"]
                summary[f"{r['method']}.peak_rss_mb"] = r["peak_rss_mb"]
        run.finish(summary)
    with pd.option_context("display.width", 160, "display.max_columns", 10):
        print(formats, sql, sep="\n")
    return summary


if __name__ == "__main__":
    run_bench()
