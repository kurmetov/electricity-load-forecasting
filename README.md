# Electricity Load Forecasting

Master's course project, topic No. 56. Day-ahead forecasting of clients' hourly electricity consumption and
search for typical daily load profiles.

- Data: [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014),
  370 clients in Portugal, a reading every 15 minutes, 2011–2014, 140,256 rows × 370 columns (51.9 million values,
  678 MB of CSV). Raw data are not in the repository; a script downloads them (menu item 1 or `python main.py download`).
- External factors: hourly temperature in Lisbon from the [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api)
  (no key) and Portuguese holidays from the `holidays` package.
- Tools: pandas, pyarrow, statsmodels, scikit-learn, Plotly, Streamlit, SQLite.
- Report: [report/report.pdf](report/report.pdf) (source `report/report.md`, `report/report.docx`).
  Slides: [report/slides.pptx](report/slides.pptx).

## Results

Day-ahead forecast, hold-out period 2014-10-01 – 2014-12-31 (92 days), every day at midnight a forecast
for 24 hours. "Total" is the summed load of all clients (kWh per hour), "clients" is the mean over 317 clients
with full data since 2012. "Time per client series" is training plus 92 daily forecasts on one client series,
the mean over 317 clients.

| Model | Total MAE | Total MAPE, % | Total MASE | Client MASE | Client sMAPE, % | Time per client series, s | Best for clients |
|---|---|---|---|---|---|---|---|
| naive | 60,035 | 33.12 | 8.57 | 3.99 | 37.2 | 0.001 | 6 |
| seasonal naive 24 h | 7,631 | 4.38 | 1.09 | 1.05 | 11.9 | 0.000 | 6 |
| seasonal naive 168 h | 11,312 | 5.78 | 1.62 | 1.08 | 10.7 | 0.000 | 15 |
| ETS (24 daily series) | 7,332 | 4.02 | 1.05 | 0.87 | 10.2 | 28.604 | 58 |
| ridge regression on lags | 6,500 | 3.86 | 0.93 | 0.90 | 11.0 | 0.030 | 30 |
| gradient boosting | **5,949** | **3.43** | **0.85** | **0.85** | **9.8** | 0.441 | **202** |

The table comes from `results/forecast/summary.csv` (run `20261117-081614_forecast` in `runs/`). Other results:
4 profile clusters (`results/clusters/`), EDA (`results/eda/`), 11 SQL queries (`results/sql/`),
time and memory comparisons (`results/bench/`), charts (`results/figures/`, interactive ones are `.html`).

## How to run

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip, see below).

```bash
git clone https://github.com/kurmetov/electricity-load-forecasting.git
cd electricity-load-forecasting
uv sync                          # dependencies from uv.lock

uv run streamlit run app.py      # dashboard: works right away on the saved results in the repository
uv run python main.py            # console menu
```

Recompute everything from scratch (downloads ~250 MB; on 32 cores the whole pipeline takes about 10 minutes):

```bash
uv run python main.py download   # or menu item 1
uv run python main.py all        # ingest -> eda -> cluster -> forecast -> db -> bench (menu item 8)
```

Single steps: `uv run python main.py eda cluster`. For a quick forecast check on a few clients:
`ELF_MAX_CLIENTS=5 uv run python main.py forecast`.

Without uv: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`, then the same commands without `uv run`.

The report and slides are rebuilt from the results (numbers are taken from the `results/` files):

```bash
uv run --group docs python report/build.py    # report.md -> report.docx -> report.pdf (needs LibreOffice)
uv run --group docs python report/slides.py   # report/slides.pptx
```

Checks: `uv run --group docs python tests/test_core.py`.

### Run log (runboard)

Each pipeline step is recorded as a run in `runs/`: config, commit, metrics, outcome. To view:
`uv run runboard` → http://127.0.0.1:5050. The plan and expectations for each step are in `docs/experiments.md`.

## Layout

```
main.py              console menu and step runner
app.py               Streamlit dashboard
elf/config.py        paths and constants
elf/data.py          download, CSV -> Parquet in chunks, float32, resampling, clock change fix
elf/external.py      weather (Open-Meteo), holidays, calendar, correlations
elf/eda.py           index, gaps, outliers, statistics, STL, ADF/KPSS, ACF/PACF, charts
elf/clustering.py    k-means of daily profiles
elf/forecast.py      baselines, ETS, ridge regression, boosting, metrics, TimeSeriesSplit, error analysis
elf/labels.py        readable names for columns and coded values
elf/db.py            SQLite: load, queries from sql/queries.sql, ER diagram
elf/bench.py         CSV vs Parquet, pandas vs SQLite: time and memory
elf/tracking.py      wrapper around runboard
sql/schema.sql       database schema;  sql/queries.sql: 11 queries
data/processed/      small derived data (daily load, clients, calendar, weather)
results/             metrics, forecasts, tables, charts
runs/                runs (runboard)
report/              report, slides and their build scripts
docs/                experiment plan, ER diagram (mermaid)
```

## Assignment requirements → where in the code

| Requirement | Where |
|---|---|
| ≥ 5 functions besides main | `elf/*.py`: `download_raw`, `convert_to_parquet`, `resample_load`, `fix_dst`, `run_eda`, `stationarity`, `run_clustering`, `predict_ets`, `metrics`, `build_db`, `run_queries`, … |
| list / dict | `MENU`, `STEPS` in `main.py`; `SLIDES` in `report/slides.py`; step results are dicts (`elf/eda.py: run_eda`), `MODEL_INFO` in `elf/db.py` |
| if / elif / else | `elf/eda.py: stationarity` (ADF/KPSS verdict), `elf/clustering.py: name_cluster`, `elf/forecast.py: fit_predict` |
| for / while | `main.py: menu` (menu loop and re-entry), `elf/data.py: convert_to_parquet` (loop over chunks), `elf/forecast.py: predict_ets` |
| read/write txt, csv, json, parquet | txt → `elf/data.py: read_raw_chunks`; parquet, csv, json → `elf/data.py`, `results/**` |
| try / except | `elf/data.py: download_raw` (URLError), `elf/external.py: fetch_weather`, `elf/db.py: run_queries` (sqlite3.OperationalError), `main.py: run_steps` (FileNotFoundError), `app.py` |
| console menu and dashboard with filters | `main.py`; `app.py` (period, all / cluster / client, series type, models) |
| several .py files with imports; os, pathlib, datetime, json | package `elf/`; `pathlib` in `elf/config.py`, `json` in `elf/data.py`, `elf/external.py`, `os` and `datetime` in `main.py` |
| comparison of ≥ 2 approaches and a baseline on quality, time and memory | models: `elf/forecast.py` (`results/forecast/summary.csv`, `memory_profile.csv`); CSV vs Parquet and pandas vs SQLite: `elf/bench.py` (`results/bench/`) |
| SQLite: load, ER diagram, ≥ 8 queries (JOIN, GROUP BY/HAVING, CTE/subqueries, window functions), results in pandas | `sql/schema.sql`, `elf/db.py`, `sql/queries.sql` (11 queries), `docs/er.md`, `results/figures/09_er.png`, `results/sql/*.csv` |
| ≥ 500 thousand rows; chunksize, type optimisation, Parquet | 140,256 × 370 = 51.9 million values; `elf/data.py: convert_to_parquet` (chunksize 20,000, float32, zstd), `load_daily` in SQLite has 540,570 rows, `to_sql(chunksize=50_000)` |
| EDA: statistics, gaps, outliers, index frequency and continuity, chunks and Parquet with memory measurement | `elf/eda.py: index_checks, missing_and_zeros, outliers`, `results/eda/`, `results/ingest.json`, `results/bench/read_formats.csv` |
| resample, moving averages, STL, ADF/KPSS, ACF/PACF | `elf/data.py: resample_load`, `elf/eda.py: make_figures, stl_strength, stationarity` |
| chronological hold-out / TimeSeriesSplit; naive and seasonal naive; ≥ 2 models | `elf/forecast.py`: hold-out from 2014-10-01, `tune` (TimeSeriesSplit), `predict_baseline`, ETS, ridge, HistGradientBoosting |
| MAE, RMSE, MAPE/sMAPE, MASE; training time; errors by period and series | `elf/forecast.py: metrics, error_breakdown`; `results/forecast/errors_*.csv`, `metrics.csv` |
| 4–5 charts, ≥ 1 interactive; weather and holidays | `results/figures/01…09`, interactive `01_daily_total.html`, `07_forecast_total.html`, the dashboard; `elf/external.py` |
| Streamlit dashboard with filters and forecast; results in Parquet/SQLite | `app.py`; `results/forecast/forecasts.parquet`, `data/interim/elf.sqlite` |
| code, data or a link, README, report, slides | this file, `report/report.pdf`, `report/slides.pptx` |

## Notes

- The timestamp in the source file is the start of a 15-minute interval: the zeros of the summer time switch fall exactly
  on 01:00–01:45. The 01:00 hour on clock-change days is fixed (`elf/data.py: fix_dst`).
- The forecast uses the actual temperature, not a weather forecast, so the quality of the models with temperature
  is slightly higher than it would be in real use.
- The SQLite database (`data/interim/elf.sqlite`, ~45 MB) and the large Parquet files are not in git; the `ingest` and `db` steps create them.
