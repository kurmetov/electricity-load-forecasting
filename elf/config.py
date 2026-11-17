"""Paths and constants shared by all modules."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"          # zip + txt, not in git
DATA_INTERIM = ROOT / "data" / "interim"  # large parquet files, not in git
DATA_PROC = ROOT / "data" / "processed"   # small derived files, in git
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SQL_DIR = ROOT / "sql"

RAW_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
RAW_ZIP = DATA_RAW / "electricityloaddiagrams20112014.zip"
RAW_TXT = DATA_RAW / "LD2011_2014.txt"
LOAD_15MIN = DATA_INTERIM / "load_15min.parquet"
LOAD_HOURLY = DATA_INTERIM / "load_hourly.parquet"
LOAD_DAILY = DATA_PROC / "load_daily.parquet"
CLIENTS = DATA_PROC / "clients.csv"
WEATHER = DATA_PROC / "weather_hourly.parquet"
CALENDAR = DATA_PROC / "calendar.parquet"
DB_PATH = DATA_INTERIM / "elf.sqlite"

USER_AGENT = "electricity-load-forecasting/1.0"
LISBON = {"latitude": 38.72, "longitude": -9.14}

START = "2011-01-01"
TRAIN_START = "2012-01-01"  # most clients have zeros in 2011 (not yet connected)
TEST_START = "2014-10-01"   # hold-out: last 3 months
END = "2014-12-31"
