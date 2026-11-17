-- SQLite schema. Daily consumption of 370 clients x 1461 days = 540 570 rows in load_daily.
DROP TABLE IF EXISTS forecast_metrics;
DROP TABLE IF EXISTS models;
DROP TABLE IF EXISTS load_daily;
DROP TABLE IF EXISTS calendar;
DROP TABLE IF EXISTS clients;
DROP TABLE IF EXISTS clusters;

CREATE TABLE clusters (
    cluster_id          INTEGER PRIMARY KEY,
    name                TEXT NOT NULL,
    peak_hour_weekday   INTEGER,
    night_to_day        REAL,
    weekend_to_weekday  REAL,
    n_clients           INTEGER
);

CREATE TABLE clients (
    client_id        TEXT PRIMARY KEY,
    first_day        TEXT,
    last_day         TEXT,
    zero_days_share  REAL,
    mean_kwh_day     REAL,
    active           INTEGER NOT NULL,
    cluster_id       INTEGER REFERENCES clusters(cluster_id)   -- NULL for inactive clients
);

CREATE TABLE calendar (
    date          TEXT PRIMARY KEY,   -- YYYY-MM-DD
    dow           INTEGER NOT NULL,   -- 0 = Monday
    is_weekend    INTEGER NOT NULL,
    is_holiday    INTEGER NOT NULL,
    holiday_name  TEXT,
    temp_mean     REAL,
    temp_min      REAL,
    temp_max      REAL
);

CREATE TABLE load_daily (
    client_id  TEXT NOT NULL REFERENCES clients(client_id),
    date       TEXT NOT NULL REFERENCES calendar(date),
    kwh        REAL NOT NULL,
    PRIMARY KEY (client_id, date)
);
CREATE INDEX idx_load_daily_date ON load_daily(date);

CREATE TABLE models (
    model        TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,       -- baseline | model
    description  TEXT
);

CREATE TABLE forecast_metrics (
    series  TEXT NOT NULL,            -- client_id or 'total'
    model   TEXT NOT NULL REFERENCES models(model),
    mae REAL, rmse REAL, mape REAL, smape REAL, mase REAL, fit_s REAL,
    PRIMARY KEY (series, model)
);
