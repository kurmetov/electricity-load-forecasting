"""Console menu. `python main.py` opens the menu; `python main.py ingest eda ...` runs steps without it."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import pandas as pd

from elf import config as cfg


def step_download():
    from elf.data import download_raw
    download_raw()


def step_ingest():
    from elf.data import ingest
    ingest()


def step_eda():
    from elf.eda import run_eda
    run_eda()


def step_cluster():
    from elf.clustering import run_clustering
    run_clustering()


def step_forecast():
    from elf.forecast import run_forecast
    limit = os.environ.get("ELF_MAX_CLIENTS")  # e.g. 5 for a quick local check
    run_forecast(max_clients=int(limit) if limit else None)


def step_db():
    from elf.db import run_db
    run_db()


def step_bench():
    from elf.bench import run_bench
    run_bench()


def show_results():
    """Print the saved results (works without the raw data)."""
    files = {
        "Ingest": cfg.RESULTS / "ingest.json",
        "Models (summary)": cfg.RESULTS / "forecast" / "summary.csv",
        "Clusters": cfg.RESULTS / "clusters" / "summary.csv",
        "CSV vs Parquet": cfg.RESULTS / "bench" / "read_formats.csv",
        "pandas vs SQLite": cfg.RESULTS / "bench" / "pandas_vs_sql.csv",
    }
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        for title, path in files.items():
            print(f"\n== {title}: {path.relative_to(cfg.ROOT)}")
            if not path.exists():
                print("   no file, run the step first")
            elif path.suffix == ".json":
                print(json.dumps(json.loads(path.read_text()), indent=2, ensure_ascii=False))
            else:
                print(pd.read_csv(path).to_string(index=False))


STEPS = {
    "download": step_download,
    "ingest": step_ingest,
    "eda": step_eda,
    "cluster": step_cluster,
    "forecast": step_forecast,
    "db": step_db,
    "bench": step_bench,
    "show": show_results,
}
PIPELINE = ["ingest", "eda", "cluster", "forecast", "db", "bench"]

MENU = [
    ("1", "Download data (UCI, ~250 MB)", ["download"]),
    ("2", "CSV -> Parquet in chunks, resample to hours and days", ["ingest"]),
    ("3", "EDA: statistics, gaps, outliers, STL, ADF/KPSS, weather and holidays", ["eda"]),
    ("4", "Profile clustering (k-means)", ["cluster"]),
    ("5", "Forecast: baselines, ETS, lag regression, boosting", ["forecast"]),
    ("6", "SQLite: load and 11 SQL queries", ["db"]),
    ("7", "Benchmarks: CSV vs Parquet, pandas vs SQL", ["bench"]),
    ("8", "Whole pipeline (2-7)", PIPELINE),
    ("9", "Show saved results", ["show"]),
    ("0", "Exit", []),
]


def run_steps(names: list[str]) -> bool:
    for name in names:
        started = datetime.now()
        print(f"\n[{started:%H:%M:%S}] {name}")
        try:
            STEPS[name]()
        except FileNotFoundError as exc:
            print(f"Missing file: {exc.filename}. Run the earlier steps first (order: {' -> '.join(PIPELINE)}).")
            return False
        print(f"[{datetime.now():%H:%M:%S}] {name}: done in {(datetime.now() - started).seconds} s")
    return True


def menu() -> None:
    actions = {key: steps for key, _, steps in MENU}
    while True:
        print("\n=== Electricity Load Forecasting ===")
        for key, title, _ in MENU:
            print(f"{key} – {title}")
        choice = input("Choose an item: ").strip()
        if choice not in actions:
            print("No such item, try again.")
            continue
        if choice == "0":
            break
        run_steps(actions[choice])


def main() -> None:
    args = sys.argv[1:]
    if not args:
        menu()
    elif unknown := [a for a in args if a not in STEPS and a != "all"]:
        sys.exit(f"unknown steps: {unknown}; available: {list(STEPS)} and all")
    else:
        ok = run_steps(PIPELINE if args == ["all"] else args)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
