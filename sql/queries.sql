-- Each query starts with "-- name: <name>". Results are saved to results/sql/<name>.csv.

-- name: q01_monthly_by_cluster
-- JOIN + GROUP BY: monthly consumption of each cluster, MWh
SELECT cl.cluster_id, cl.name, substr(l.date, 1, 7) AS month,
       ROUND(SUM(l.kwh) / 1000, 1) AS mwh, COUNT(DISTINCT l.client_id) AS clients
FROM load_daily l
JOIN clients c  ON c.client_id = l.client_id
JOIN clusters cl ON cl.cluster_id = c.cluster_id
WHERE l.date >= '2012-01-01'
GROUP BY cl.cluster_id, cl.name, month
ORDER BY month, cl.cluster_id;

-- name: q02_large_stable_clients
-- GROUP BY + HAVING: large clients (> 5 MWh per day on average) with almost no zero days since 2012
SELECT client_id, ROUND(AVG(kwh), 1) AS avg_kwh_day,
       SUM(CASE WHEN kwh = 0 THEN 1 ELSE 0 END) AS zero_days
FROM load_daily
WHERE date >= '2012-01-01'
GROUP BY client_id
HAVING AVG(kwh) > 5000 AND SUM(CASE WHEN kwh = 0 THEN 1 ELSE 0 END) < 10
ORDER BY avg_kwh_day DESC;

-- name: q03_weekend_ratio_cte
-- CTE + JOIN: weekend / weekday consumption ratio per cluster
WITH per_client AS (
    SELECT l.client_id,
           AVG(CASE WHEN k.is_weekend = 0 AND k.is_holiday = 0 THEN l.kwh END) AS weekday_kwh,
           AVG(CASE WHEN k.is_weekend = 1 THEN l.kwh END) AS weekend_kwh
    FROM load_daily l
    JOIN calendar k ON k.date = l.date
    WHERE l.date >= '2012-01-01'
    GROUP BY l.client_id
)
SELECT cl.cluster_id, cl.name, COUNT(*) AS clients,
       ROUND(AVG(p.weekend_kwh / p.weekday_kwh), 3) AS mean_weekend_ratio,
       ROUND(MIN(p.weekend_kwh / p.weekday_kwh), 3) AS min_ratio,
       ROUND(MAX(p.weekend_kwh / p.weekday_kwh), 3) AS max_ratio
FROM per_client p
JOIN clients c   ON c.client_id = p.client_id
JOIN clusters cl ON cl.cluster_id = c.cluster_id
GROUP BY cl.cluster_id, cl.name
ORDER BY cl.cluster_id;

-- name: q04_rolling_7d_total
-- Window function: total daily load and its 7-day and 28-day moving averages
WITH total AS (
    SELECT date, SUM(kwh) / 1000 AS mwh FROM load_daily GROUP BY date
)
SELECT date, ROUND(mwh, 1) AS mwh,
       ROUND(AVG(mwh) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS ma7,
       ROUND(AVG(mwh) OVER (ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW), 1) AS ma28
FROM total
WHERE date >= '2012-01-01'
ORDER BY date;

-- name: q05_top3_in_cluster
-- Window RANK in a subquery: three largest clients of each cluster in 2014
SELECT * FROM (
    SELECT c.cluster_id, l.client_id, ROUND(SUM(l.kwh) / 1000, 1) AS mwh_2014,
           RANK() OVER (PARTITION BY c.cluster_id ORDER BY SUM(l.kwh) DESC) AS rnk
    FROM load_daily l
    JOIN clients c ON c.client_id = l.client_id
    WHERE l.date BETWEEN '2014-01-01' AND '2014-12-31' AND c.cluster_id IS NOT NULL
    GROUP BY c.cluster_id, l.client_id
)
WHERE rnk <= 3
ORDER BY cluster_id, rnk;

-- name: q06_peak_days_subquery
-- Subquery: days whose total load is 10 % above the 2012-2014 average
SELECT t.date, k.dow, k.holiday_name, k.temp_mean, ROUND(t.mwh, 1) AS mwh
FROM (SELECT date, SUM(kwh) / 1000 AS mwh FROM load_daily GROUP BY date) t
JOIN calendar k ON k.date = t.date
WHERE t.date >= '2012-01-01'
  AND t.mwh > 1.10 * (SELECT AVG(mwh) FROM (SELECT SUM(kwh) / 1000 AS mwh FROM load_daily
                                             WHERE date >= '2012-01-01' GROUP BY date))
ORDER BY t.mwh DESC;

-- name: q07_holiday_effect
-- JOIN calendar: average client day on working days, Saturdays/Sundays and public holidays, per cluster
SELECT cl.name,
       ROUND(AVG(CASE WHEN k.is_holiday = 0 AND k.is_weekend = 0 THEN l.kwh END), 1) AS workday_kwh,
       ROUND(AVG(CASE WHEN k.is_holiday = 0 AND k.is_weekend = 1 THEN l.kwh END), 1) AS weekend_kwh,
       ROUND(AVG(CASE WHEN k.is_holiday = 1 THEN l.kwh END), 1) AS holiday_kwh
FROM load_daily l
JOIN calendar k  ON k.date = l.date
JOIN clients c   ON c.client_id = l.client_id
JOIN clusters cl ON cl.cluster_id = c.cluster_id
WHERE l.date >= '2012-01-01'
GROUP BY cl.cluster_id, cl.name
ORDER BY cl.cluster_id;

-- name: q08_yearly_growth_lag
-- CTE + LAG: change of yearly consumption 2012 -> 2013 -> 2014 per client, largest changes in 2014
WITH yearly AS (
    SELECT client_id, substr(date, 1, 4) AS year, SUM(kwh) / 1000 AS mwh
    FROM load_daily
    WHERE date >= '2012-01-01'
    GROUP BY client_id, year
), growth AS (
    SELECT client_id, year, mwh,
           LAG(mwh) OVER (PARTITION BY client_id ORDER BY year) AS prev_mwh
    FROM yearly
)
SELECT client_id, year, ROUND(prev_mwh, 1) AS prev_mwh, ROUND(mwh, 1) AS mwh,
       ROUND(100.0 * (mwh - prev_mwh) / prev_mwh, 1) AS growth_pct
FROM growth
WHERE year = '2014' AND prev_mwh > 100
ORDER BY ABS(growth_pct) DESC
LIMIT 15;

-- name: q09_load_by_temperature
-- JOIN + CASE + GROUP BY: total load on working days by temperature band
WITH total AS (
    SELECT date, SUM(kwh) / 1000 AS mwh FROM load_daily GROUP BY date
)
SELECT CASE WHEN k.temp_mean < 10 THEN '1: < 10'
            WHEN k.temp_mean < 14 THEN '2: 10-14'
            WHEN k.temp_mean < 18 THEN '3: 14-18'
            WHEN k.temp_mean < 22 THEN '4: 18-22'
            ELSE '5: >= 22' END AS temp_band,
       COUNT(*) AS days, ROUND(AVG(t.mwh), 1) AS avg_mwh
FROM total t
JOIN calendar k ON k.date = t.date
WHERE t.date >= '2012-01-01' AND k.is_weekend = 0 AND k.is_holiday = 0
GROUP BY temp_band
ORDER BY temp_band;

-- name: q10_best_model_per_series
-- ROW_NUMBER: best model by MASE for every client, then wins per model
WITH ranked AS (
    SELECT series, model, mase,
           ROW_NUMBER() OVER (PARTITION BY series ORDER BY mase) AS rn
    FROM forecast_metrics
    WHERE series <> 'total'
)
SELECT r.model, m.kind, COUNT(*) AS wins, ROUND(AVG(r.mase), 3) AS mean_mase_when_best
FROM ranked r
JOIN models m ON m.model = r.model
WHERE r.rn = 1
GROUP BY r.model, m.kind
ORDER BY wins DESC;

-- name: q11_quality_by_cluster
-- JOIN + GROUP BY + HAVING: mean MASE and sMAPE of each model inside each cluster
SELECT cl.name, f.model, COUNT(*) AS clients,
       ROUND(AVG(f.mase), 3) AS mean_mase, ROUND(AVG(f.smape), 2) AS mean_smape
FROM forecast_metrics f
JOIN clients c   ON c.client_id = f.series
JOIN clusters cl ON cl.cluster_id = c.cluster_id
GROUP BY cl.cluster_id, cl.name, f.model
HAVING COUNT(*) >= 5
ORDER BY cl.cluster_id, mean_mase;
