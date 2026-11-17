# Experiments

Each pipeline step is a separate run in `runs/` (tag `step`). For every step below: what is checked
and what was expected before the run. After the run, the run id and the outcome are added here.

Data: UCI ElectricityLoadDiagrams20112014, 370 clients, 15 min, 2011-01-01 – 2014-12-31.
Training from 2012-01-01 (many clients are not connected yet in 2011), hold-out period
2014-10-01 – 2014-12-31 (92 days, 2,208 hours).

## S1 ingest

Question: how much memory and disk space do chunked reading, float32 and Parquet save.
Expectation: float32 takes half the memory of float64; Parquet (zstd) is 5–10 times smaller than CSV on disk.

## S2 eda

Question: index quality (frequency, duplicates, gaps), missing values and zeros, outliers, seasonality,
stationarity, relation to temperature and holidays.
Expectation: continuous 15-minute index; no NaN, leading zeros for clients connected later;
outliers on DST change days; strong daily and weekly seasonality; the daily series is non-stationary
by KPSS because of trend and yearly seasonality; load on holidays is lower than on weekdays.

## S3 cluster

Question: which typical daily profiles the clients have and how many.
Method: k-means on 48 features (mean weekday and weekend hour, normalised by the client's mean),
k = 2..10, k chosen by silhouette among k >= 3.
Expectation: 3–5 clusters: office (weekdays, daytime), household (evening peak), around the clock.

## S4 forecast

Question: how much better ETS, ridge regression on lags and gradient boosting are than naive
day-ahead forecasts.
Setup: on each hold-out day at midnight a forecast for 24 hours; only features with lag >= 24 h;
ridge and boosting hyperparameters are tuned with TimeSeriesSplit (4 folds of 28 days) on the training part
of the total series.
Expectation: seasonal naive 168 h better than 24 h; ridge and boosting 15–30 % better than seasonal naive by MASE;
ETS in between; higher errors on holidays (December) and on clients with irregular load.

## S5 sql

Question: loading the derived tables into SQLite and 11 queries (JOIN, GROUP BY/HAVING, CTE,
subqueries, window functions) with results in pandas.
Expectation: load_daily = 370 × 1,461 = 540,570 rows; every query runs in under a second.

## S6 bench

Question: time and memory of reading CSV (default and in chunks with float32) vs Parquet;
pandas aggregation vs SQLite.
Expectation: Parquet is read 10+ times faster than CSV; SQLite is comparable to pandas on daily aggregation.

## Run outcomes

The final set is all steps on commit b9d5e17 (`python main.py all`). The report takes its numbers from these runs.

| Step | Run | Outcome vs expectation |
|---|---|---|
| S1 | 20261117-085143_ingest | float32 is half of float64 (28.4 vs 56.6 MB per chunk), as expected. Parquet 41.6 MB vs 678 MB of CSV, 16 times smaller, more than expected. |
| S2 | 20261117-085146_eda | Continuous index, no NaN, leading zeros present, as expected. Weekly seasonality is weak (0.07), a strong one was expected. Load grows with temperature (r = 0.90): cooling, not heating. Holiday −11.4 %. |
| S3 | 20261117-085239_cluster | k = 4, silhouette 0.423. Profiles: evening, flat, day and evening, weekday. With k ≥ 5 a cluster of one client appears. |
| S4 | 20261117-085309_forecast | Boosting is the best: total MASE 0.85, best for 202 of 317 clients. Seasonal naive 24 h beats 168 h on the total, contrary to expectation (the weekly cycle is weak). Errors on holidays are ~3 times higher. |
| S5 | 20261117-085940_sql | load_daily has 540,570 rows, 11 queries, all < 0.3 s. |
| S6 | 20261117-085942_bench | Parquet is read 18 times faster than CSV, but peak memory when reading the whole file is not lower. SQLite is slower than pandas (0.17 vs 0.07 s) but uses 12 times less memory. |

Intermediate runs (their recorded commits come from the project history before the English translation and are not in this repository):

- 20261116-195315_forecast: ETS as one model on the hourly series (168 h season, 56-day window): total MAE 20,963 vs 7,631 for seasonal naive 24 h. The level follows the last evening hours. Replaced by ETS on 24 daily series, settings chosen on September 2014.
- 20261116-195734_forecast: the same model code as the final run; the metrics match.
- 20261116-200504_bench, 20261116-200553_bench: memory was measured wrongly (tracemalloc does not see pyarrow, then ru_maxrss without resetting the peak). Fixed: peak RSS is reset via /proc/self/clear_refs.
- 20261116-195010_ingest, 195013_eda, 195220_cluster, 200501_sql, 200626_bench: early versions of the same steps.
