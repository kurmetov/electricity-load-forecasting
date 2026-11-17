# Electricity Load Forecasting and Typical Load Profiles

Course project, topic No. 56 "Electricity Load Forecasting". Code: the `electricity-load-forecasting` repository; how to run it is in README.md.

## 1. Task

Using the consumption history of an electricity supplier's clients, build a day-ahead forecast of hourly load and find typical daily consumption profiles. The forecast is compared with simple baselines (naive and seasonal naive) on accuracy, training time and memory. Profiles are found with k-means clustering.

## 2. Data

The source is the UCI ElectricityLoadDiagrams20112014 dataset. It is one text file, `LD2011_2014.txt`: separator `;`, decimal comma, the first column is the timestamp, followed by 370 columns `MT_001` … `MT_370`. Values are mean power in kW over 15 minutes, so they are divided by 4 to get kWh. The file is {{json:results/ingest.json:convert.csv_mb|.0f}} MB, {{json:results/ingest.json:convert.rows|,.0f}} rows, {{json:results/ingest.json:convert.values|,.0f}} values in total.

External data:

- hourly air temperature at 2 m in Lisbon (38.72° N, 9.14° W) for 2011–2014 from the Open-Meteo archive, requested without a key and without personal data;
- Portuguese public holidays from the `holidays` package. Some holidays were cancelled in 2013–2014, and the package takes this into account.

These two sources give a calendar for every day: day of week, weekend, holiday, mean, minimum and maximum temperature (`data/processed/calendar.parquet`).

## 3. Data preparation

The file is read by `read_raw_chunks` in chunks of 20,000 rows. Each chunk is converted from float64 to float32 at once and appended to Parquet with zstd compression (`convert_to_parquet`). One chunk takes {{json:results/ingest.json:convert.chunk_mb_float64|.1f}} MB in float64 and {{json:results/ingest.json:convert.chunk_mb_float32|.1f}} MB in float32. The whole dataset in float32 takes {{json:results/ingest.json:convert.full_mb_float32_est|.0f}} MB of memory instead of {{json:results/ingest.json:convert.full_mb_float64_est|.0f}} MB, and Parquet on disk takes {{json:results/ingest.json:convert.parquet_mb|.1f}} MB instead of {{json:results/ingest.json:convert.csv_mb|.0f}} MB of CSV. The conversion took {{json:results/ingest.json:convert.seconds}} s.

Ways to read the same data (`elf/bench.py`; time is the median of several runs, memory is the growth of peak RSS of a separate process):

{{csv:results/bench/read_formats.csv|method,file_mb,seconds,peak_rss_mb,frame_mb}}

Parquet is read in {{cell:results/bench/read_formats.csv:method=parquet zstd, float32:seconds|.3f}} s against {{cell:results/bench/read_formats.csv:method=csv, pandas default (float64):seconds|.2f}} s for CSV (ratio {{json:results/report/ratios.json:csv_over_parquet_time|.0f}}) and takes {{json:results/ingest.json:convert.parquet_mb|.1f}} MB on disk against {{json:results/ingest.json:convert.csv_mb|.0f}} MB. CSV is also read fast, because pandas 3 parses it in C and after the first read the file sits in the OS cache; on a cold read the gap is larger. There is no gain in peak memory: pyarrow first builds the table in its own format and then copies it into pandas. Only streaming saves memory, when each chunk is reduced to daily sums at once and only one chunk is held in memory. Reading 10 columns from Parquet is almost free, which is the main advantage of a columnar format.

The data are then aggregated to hours and days (`resample_load`). The timestamp in the file is the start of a 15-minute interval. This is visible on the days of the switch to summer time: all clients have zeros exactly at 01:00–01:45. The file ends with the timestamp 2015-01-01 00:00; this last incomplete hour is dropped. The result is 35,064 hours and 1461 days.

On the clock-change days (last Sunday of March and October) in 2012–2014 there is an artefact: in March the 01:00 hour is zero, in October it holds two hours. There is no artefact in 2011. The function `fix_dst` checks the total load and, if the artefact is present, replaces the March hour with the mean of its neighbours and halves the October hour.

## 4. Exploratory analysis

**Index.** The step is 15 minutes everywhere ({{json:results/eda/eda.json:index.inferred_freq}}), the index is monotonic, duplicates: {{json:results/eda/eda.json:index.duplicates}}, missing timestamps: {{json:results/eda/eda.json:index.missing_steps}}.

**Gaps.** There are no NaN values ({{json:results/eda/eda.json:missing_zeros.nan_values}}). Gaps are coded as zeros: the client is not connected yet or already disconnected. {{json:results/eda/eda.json:missing_zeros.clients_by_first_year.2011}} clients have data from 2011, another {{json:results/eda/eda.json:missing_zeros.clients_by_first_year.2012}} start in 2012 (mostly on 1 January), {{json:results/eda/eda.json:missing_zeros.clients_by_first_year.2013}} in 2013 and {{json:results/eda/eda.json:missing_zeros.clients_by_first_year.2014}} in 2014. So training starts on 2012-01-01, and forecasting and clustering use {{json:results/eda/eda.json:missing_zeros.clients_active}} clients that have consumption from 2012-01-01 to 2014-12-31 and less than 5 % zero days.

**Descriptive statistics** of the total load, kWh (2012–2014):

{{csv:results/eda/total_stats.csv}}

Statistics for each client are in `results/eda/client_stats.csv`. Clients differ a lot in size: the median of mean daily consumption is orders of magnitude below the maximum, so comparing models across clients needs scale-free metrics (MASE, sMAPE).

**Outliers.** For the daily consumption of each active client a robust z-score |x − median| / (1.4826 · MAD) was computed; a value above 5 counts as an outlier. {{json:results/eda/eda.json:outliers.flagged_values|,.0f}} client-days ({{json:results/eda/eda.json:outliers.flagged_share_pct}} %) of {{json:results/eda/eda.json:outliers.clients_with_flags}} clients were flagged. Most of them are 25 December and 1 January, when dozens of clients have outliers at once. These are not measurement errors but real drops in load, so the values were kept and holidays were added to the model features.

![Fig. 1. Total daily consumption and 7- and 30-day moving averages](results/figures/01_daily_total.png)

Fig. 1 shows the yearly seasonality: in July–August consumption is about 45 % higher than in January–February. The sharp dips are Christmas and New Year. An interactive version of the chart is `results/figures/01_daily_total.html`.

**STL and seasonality.** STL decomposition of the daily series with period 7 gives a trend strength of {{json:results/eda/eda.json:stl_daily.trend_strength}} and a weekly seasonal strength of only {{json:results/eda/eda.json:stl_daily.seasonal_strength}}. For the 2014 hourly series with period 24 the daily seasonal strength is {{json:results/eda/eda.json:stl_hourly_2014.seasonal_strength}}. So the total load has a very strong daily cycle and almost no weekly one: weekends differ little from weekdays (Fig. 4).

![Fig. 2. STL decomposition of the total daily series](results/figures/02_stl_daily.png)

**Stationarity.** ADF test (null hypothesis: unit root) and KPSS test (null hypothesis: stationarity):

{{csv:results/eda/stationarity.csv}}

The raw daily series is non-stationary by ADF (p = {{json:results/eda/eda.json:stationarity.0.adf_p}}) and borderline by KPSS. After the first difference or the 7-day difference both tests agree that the series is stationary. For the hourly series the 24-hour difference does the same. This is why the regression uses lags and means over the previous day and week rather than the levels themselves.

![Fig. 3. ACF and PACF of the hourly and daily series](results/figures/03_acf_pacf.png)

The ACF of the hourly series has peaks at 24, 48, 72, … hours and decays slowly, which confirms the daily seasonality. In the PACF lags 1, 2 and 23–25 hours are significant. Lags shorter than 24 hours cannot be used in a day-ahead forecast, so the model uses lags of 24, 48, 72, 168 and 336 hours.

![Fig. 4. Mean hourly load by day of week, MWh](results/figures/04_heatmap_hour_dow.png)

## 5. Weather and holidays

![Fig. 5. Daily consumption and mean temperature](results/figures/05_temp_vs_load.png)

Correlation of the total daily load with mean temperature: Pearson {{json:results/eda/eda.json:external.pearson_temp|.3f}}, Spearman {{json:results/eda/eda.json:external.spearman_temp|.3f}}. The relation is non-linear: up to 15 °C the load is almost flat, above that it grows. This is cooling; electric heating is hardly visible for these clients (the correlation with cooling degree-days above 22 °C is {{json:results/eda/eda.json:external.pearson_cdd|.2f}}, and with heating degree-days below 16 °C it is {{json:results/eda/eda.json:external.pearson_hdd|.2f}}, so in cold weather consumption is even lower).

Mean working-day load by temperature band (SQL query `q09`):

{{csv:results/sql/q09_load_by_temperature.csv}}

A holiday on a weekday lowers the load by {{json:results/eda/eda.json:external.holiday_drop_pct|.1f}} % on average ({{json:results/eda/eda.json:external.holiday_mean_mwh|,.0f}} MWh against {{json:results/eda/eda.json:external.workday_mean_mwh|,.0f}} MWh on a normal weekday; there were {{json:results/eda/eda.json:external.n_holidays_on_workdays|.0f}} such days in 2012–2014). Part of the effect comes from the season: many holidays are in winter, when the load is lower anyway. Weekends hardly differ from weekdays: {{json:results/eda/eda.json:external.weekend_mean_mwh|,.0f}} MWh.

## 6. Typical load profiles

For each of the {{json:results/clusters/result.json:clients}} clients, hourly means were computed on 2012-01-01 – 2014-09-30 (without the test period) separately for weekdays and weekends, 48 numbers. The vector is divided by the client's mean, so the clusters reflect the shape of the profile, not the size of the client.

Clustering is k-means (scikit-learn, 20 runs from different initial centres) for k from 2 to 10:

{{csv:results/clusters/k_selection.csv}}

The silhouette grows up to k = 8, but from k = 5 on there is a cluster of one client, and such a cluster is an outlier, not a typical profile. The rule: the highest silhouette among k ≥ 3 where every cluster has at least 5 clients. This gives k = {{json:results/clusters/result.json:k}}, silhouette {{json:results/clusters/result.json:silhouette}}.

![Fig. 6. Cluster centres (share of the client's mean)](results/figures/06_cluster_profiles.png)

{{csv:results/clusters/summary.csv|cluster,name,n_clients,peak_hour_weekday,night_to_day,weekend_to_weekday}}

- Cluster 0, "evening peak, moderate night", is the largest. Load rises in the morning, stays flat during the day and peaks at 18:00; at night it is about half of the daytime level; weekends look like weekdays.
- Cluster 1, "flat around the clock": almost as much at night as during the day, a small evening peak at 20:00.
- Cluster 2, "day and evening, near zero at night": work from 9:00 to 22:00 without days off, at night about a quarter of the daytime level. These are large clients, about 55 MWh per day on average (query `q07`).
- Cluster 3, "weekday daytime": a daytime peak on weekdays, at weekends consumption is half as high (ratio {{cell:results/clusters/summary.csv:cluster=3:weekend_to_weekday}}). It looks like offices and manufacturing.

The clients are anonymous, so the names describe only the shape of the profile.

## 7. Forecasting methods

**Setup.** Time-based hold-out: training 2012-01-01 – 2014-09-30, test 2014-10-01 – 2014-12-31 (92 days, 2208 hours). On each test day at midnight a forecast for the next 24 hours is made using only data before that midnight. Forecasts are made for the total load and for each of the {{json:results/forecast/params.json:n_clients}} clients separately, {{json:results/forecast/params.json:n_series}} series in total. Series are processed in parallel (joblib).

**Baselines:**

- naive: the last known value (hour 23 of the previous day) for all 24 hours;
- seasonal naive 24 h: the value of the same hour yesterday;
- seasonal naive 168 h: the value of the same hour a week ago.

**ETS.** The first version was one Holt–Winters model on the hourly series with a 168-hour season. It lost to the seasonal naive 24 h forecast: total MAE 20,963 against 7631 kWh. The level smoothing came out around 0.9, and the level of the last evening hours was carried over to the whole next day. In the second version the hourly series is cut into 24 daily series (all values at 00:00, all at 01:00 and so on), and each gets simple exponential smoothing with an additive weekly season (statsmodels `ExponentialSmoothing`), refitted on every test day.

The window and the season were chosen by MAE on September 2014, the last month of the training part; the test was not used for this:

{{csv:results/forecast/tuning_ets.csv}}

The chosen window is {{json:results/forecast/params.json:ets_window_days}} days with a weekly season.

**Ridge regression on lags** and **gradient boosting** (scikit-learn `HistGradientBoostingRegressor`). Features: lags of 24, 48, 72, 168 and 336 hours, means over 24 and 168 hours up to lag 24, hour, day of week, month, holiday flag, temperature and the square of its deviation from 18 °C. All lags are at least 24 hours, so every hour of the next day is known at midnight. For ridge regression, hour, day of week and month are one-hot encoded and numeric features are standardised. Each model is trained once on the training part of each series.

Hyperparameters were tuned on the total series with `TimeSeriesSplit` (4 folds of 28 days, windows only expand forward, no shuffling):

{{csv:results/forecast/tuning_cv.csv|model,ridge_alpha,gbm_lr,gbm_leaves,cv_mae,cv_mae_std}}

Chosen: ridge regularisation alpha = {{json:results/forecast/params.json:ridge_alpha}}; for boosting, learning rate {{json:results/forecast/params.json:gbm_lr}}, at most {{json:results/forecast/params.json:gbm_leaves}} leaves per tree, 300 iterations.

**Metrics:** MAE, RMSE, MAPE (only over hours with non-zero actuals), sMAPE and MASE. The MASE denominator is the MAE of the seasonal naive 24 h forecast on the training part of the same series, so MASE < 1 means "better than yesterday's values on the history".

## 8. Forecast results

Total load (kWh per hour) and the mean over clients:

{{csv:results/forecast/summary.csv|model,total_mae,total_rmse,total_mape,total_mase,mase_mean,mase_median,smape_mean,wins_mase}}

The first four numeric columns are for the total load, the rest are over clients; "best for clients" is the number of clients for which the model had the lowest MASE.

Time and memory: time is training plus forecasting one series over all 92 test days (for clients, the mean over series); memory is the `tracemalloc` peak on the total series:

{{csv:results/report/time_memory.csv}}

Main points:

- Boosting is the best both on the total and over clients: total MAE {{cell:results/forecast/summary.csv:model=gbm:total_mae|,.0f}} kWh, {{json:results/report/ratios.json:gbm_vs_snaive24_pct|.0f}} % lower than seasonal naive 24 h; it is the best for {{cell:results/forecast/summary.csv:model=gbm:wins_mase}} of {{json:results/forecast/params.json:n_clients}} clients.
- Ridge regression is second on the total (MAE {{cell:results/forecast/summary.csv:model=ridge:total_mae|,.0f}}), but over clients it is slightly worse than ETS on average. It is the fastest of the trained models: {{cell:results/forecast/summary.csv:model=ridge:fit_s_mean|.3f}} s per series.
- ETS on 24 daily series beats seasonal naive, by a small margin on the total (MAE {{cell:results/forecast/summary.csv:model=ets:total_mae|,.0f}}) and more clearly over clients (mean MASE {{cell:results/forecast/summary.csv:model=ets:mase_mean|.3f}} against {{cell:results/forecast/summary.csv:model=snaive24:mase_mean|.3f}}). It is the slowest method: {{cell:results/forecast/summary.csv:model=ets:fit_s_mean|.1f}} s per series, because 24 models are fitted for every test day.
- For the total, seasonal naive 24 h is better than 168 h: the weekly seasonality is weak, and the weather changes more over a week than over a day. For clients of the weekday cluster it is the other way round, 168 h is better (see below).
- The naive forecast (last value) is useless for hourly forecasting: MASE {{cell:results/forecast/summary.csv:model=naive:total_mase|.1f}}.

![Fig. 7. Forecast of the total load, 15–28 December 2014](results/figures/07_forecast_total.png)

An interactive version with all test days is `results/figures/07_forecast_total.html`.

**Errors by period.** sMAPE by month and by day type (mean over all series and hours):

{{csv:results/forecast/errors_by_month.csv}}

{{csv:results/forecast/errors_by_day_type.csv}}

December is worse than October and November for all models. The main reason is holidays: the error on holidays is about three times higher than on working days. The worst days for boosting:

{{csv:results/forecast/errors_worst_days_gbm.csv}}

On 25 December the models expected a normal day (Fig. 7), but the load fell almost by half. The holiday flag is among the features, but the training data contain only 2–3 Christmases, and boosting did not learn it. On 26 December ridge regression and ETS erred the other way: they rely on the 24-hour lag, which is exactly the Christmas dip.

![Fig. 8. sMAPE by month and by hour of day](results/figures/08_errors.png)

By hour of day: ETS and boosting are best at night and worst during the day from 10:00 to 17:00, when the load depends more on the weather.

**Errors by cluster** (median MASE over the clients of a cluster):

{{csv:results/forecast/errors_mase_by_cluster.csv}}

The weekday daytime cluster 3 is the easiest to forecast: it has a stable weekly rhythm, and even seasonal naive 168 h gives MASE {{cell:results/forecast/errors_mase_by_cluster.csv:cluster=3:snaive168|.2f}}. Clusters 0 and 2 are the hardest: the median MASE of boosting is {{cell:results/forecast/errors_mase_by_cluster.csv:cluster=0:gbm|.2f}} and {{cell:results/forecast/errors_mase_by_cluster.csv:cluster=2:gbm|.2f}}, about 13 % better than seasonal naive 24 h.

## 9. SQLite database

Derived data are loaded into SQLite (`elf/db.py`, schema `sql/schema.sql`). The `load_daily` table is written with `to_sql` in chunks of 50,000 rows.

![Fig. 9. ER diagram](results/figures/09_er.png)

Rows per table:

| Table | Rows |
|---|---|
| `clusters` | {{json:results/sql/db.json:rows.clusters}} |
| `clients` | {{json:results/sql/db.json:rows.clients}} |
| `calendar` | {{json:results/sql/db.json:rows.calendar}} |
| `load_daily` | {{json:results/sql/db.json:rows.load_daily|,.0f}} |
| `models` | {{json:results/sql/db.json:rows.models}} |
| `forecast_metrics` | {{json:results/sql/db.json:rows.forecast_metrics}} |

The queries are in `sql/queries.sql`, run through `pandas.read_sql_query`, and the results are saved to `results/sql/`:

| Query | What it does | SQL features |
|---|---|---|
| `q01_monthly_by_cluster` | cluster consumption by month | JOIN of three tables, GROUP BY |
| `q02_large_stable_clients` | large clients with almost no zero days | GROUP BY, HAVING |
| `q03_weekend_ratio_cte` | weekend/weekday ratio by cluster | CTE, JOIN, CASE |
| `q04_rolling_7d_total` | 7- and 28-day moving averages | CTE, window AVG OVER (ROWS BETWEEN) |
| `q05_top3_in_cluster` | three largest clients of each cluster in 2014 | RANK() OVER (PARTITION BY), subquery |
| `q06_peak_days_subquery` | days 10 % above the mean | subqueries in FROM and WHERE, JOIN |
| `q07_holiday_effect` | a client's mean day on weekdays, weekends and holidays | JOIN, CASE, GROUP BY |
| `q08_yearly_growth_lag` | change of a client's yearly consumption | CTE, LAG() OVER |
| `q09_load_by_temperature` | load by temperature band | CTE, CASE, GROUP BY |
| `q10_best_model_per_series` | best model for each client | ROW_NUMBER() OVER, CTE, JOIN |
| `q11_quality_by_cluster` | model quality within clusters | JOIN, GROUP BY, HAVING |

For example, `q10` confirms the pandas count: boosting is the best model for {{cell:results/sql/q10_best_model_per_series.csv:model=gbm:wins}} clients. Query times are in `results/sql/timing.csv`; all are under 0.3 s.

**pandas against SQLite.** The same aggregation, cluster consumption by month since 2012:

{{csv:results/bench/pandas_vs_sql.csv|method,seconds,peak_rss_mb,result_rows,same_result}}

The results match. pandas is faster, because the daily table in Parquet is small and fits in memory, but it needs about {{json:results/report/ratios.json:pandas_over_sql_memory|.0f}} times more memory: the table is unpivoted into a long format of 400 thousand rows. SQLite reads only what it needs from disk and uses almost no memory.

## 10. Dashboard

`streamlit run app.py` opens the dashboard. Filters: period, object (all clients, a cluster or one client), series type (daily, 7- or 30-day moving average), forecast models. Tabs: consumption for the period, hourly forecast on the test period with metrics for the chosen object, cluster centres, model comparison by MASE. The dashboard reads only saved files (`data/processed/`, `results/`), so it starts right after cloning the repository.

![Fig. 10. Dashboard: forecast for a cluster](report/dashboard.png)

## 11. Conclusions

1. The data are of good quality: the index has no gaps or duplicates, there are no NaN values. The problems are zeros before a client is connected and the clock-change artefact; both are handled.
2. Reading in chunks with float32 and storing in Parquet shrink the file from {{json:results/ingest.json:convert.csv_mb|.0f}} to {{json:results/ingest.json:convert.parquet_mb|.0f}} MB and the read time from {{cell:results/bench/read_formats.csv:method=csv, pandas default (float64):seconds|.2f}} to {{cell:results/bench/read_formats.csv:method=parquet zstd, float32:seconds|.2f}} s. Only streaming over chunks reduces peak memory.
3. The total load has a strong daily cycle, a weak weekly cycle and a yearly cycle tied to temperature: in summer the load is higher because of cooling.
4. Four typical profiles were found: evening (168 clients), flat around the clock (95), day without night (33) and weekday office (21).
5. Gradient boosting on lags, calendar and temperature is the best model: total MASE {{cell:results/forecast/summary.csv:model=gbm:total_mase|.2f}}, best for {{cell:results/forecast/summary.csv:model=gbm:wins_mase}} of {{json:results/forecast/params.json:n_clients}} clients. Ridge regression is slightly worse but {{json:results/report/ratios.json:gbm_over_ridge_time|.0f}} times faster. ETS beats seasonal naive only in the version with 24 daily series and is the slowest.
6. Most errors fall on holidays and the days before them.

**Limitations.** The models get the actual temperature, while a real forecast would use a forecast temperature, so the scores of ridge regression and boosting are slightly optimistic. One weather point (Lisbon) for the whole country. The test covers only three months and a few holidays. The clients are anonymous, so the client type can only be guessed from the cluster.

## 12. Reproducibility

Every step is recorded as a run in `runs/` (viewed with `runboard`): config, commit, metrics. The numbers in this report are filled in by `report/build.py` from the `results/` files written by the runs:

{{csv:results/report/runs.csv}}
