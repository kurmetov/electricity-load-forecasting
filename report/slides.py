"""Slides: report/slides.pptx. Numbers come from results files through the same placeholders as the report.
Run: uv run --group docs python report/slides.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import ROOT, csv_table, derive, resolve  # noqa: E402

FIG = "results/figures/"
SLIDES = [
    {"title": "Electricity Load Forecasting and Typical Load Profiles",
     "subtitle": "Topic No. 56 · UCI ElectricityLoadDiagrams 2011–2014 · pandas, statsmodels, scikit-learn, Plotly"},
    {"title": "Task and data",
     "bullets": [
         "Day-ahead forecast of hourly consumption and search for typical daily client profiles",
         "370 clients in Portugal, a reading every 15 minutes, 2011-01-01 – 2014-12-31: "
         "{{json:results/ingest.json:convert.rows|,.0f}} rows × 370 = {{json:results/ingest.json:convert.values|,.0f}} values, "
         "{{json:results/ingest.json:convert.csv_mb|.0f}} MB of CSV",
         "Values in kW per 15 min → kWh (÷4), resampled to hours and days",
         "External factors: temperature in Lisbon (Open-Meteo archive), Portuguese holidays (holidays package)",
         "Training 2012-01-01 – 2014-09-30, hold-out 2014-10-01 – 2014-12-31 (92 days)",
     ]},
    {"title": "Code layout",
     "bullets": [
         "main.py: console menu and step runner, python main.py all",
         "elf/data.py: download, CSV → Parquet in chunks, float32, resampling, DST fix",
         "elf/eda.py, elf/external.py: EDA, STL, ADF/KPSS, ACF/PACF, weather and holidays",
         "elf/clustering.py: k-means of profiles; elf/forecast.py: baselines, ETS, ridge regression, boosting, metrics",
         "elf/db.py + sql/: SQLite, 11 queries, ER diagram; elf/bench.py: time and memory comparison",
         "app.py: Streamlit dashboard; elf/tracking.py: every step is recorded as a run in runboard (runs/)",
     ]},
    {"title": "Ingest: chunks, float32, Parquet",
     "table": "results/bench/read_formats.csv|method,file_mb,seconds,peak_rss_mb,frame_mb",
     "note": "Parquet (zstd) takes {{json:results/ingest.json:convert.parquet_mb|.0f}} MB instead of {{json:results/ingest.json:convert.csv_mb|.0f}} MB of CSV "
             "and is read {{json:results/report/ratios.json:csv_over_parquet_time|.0f}} times faster. float32 halves the table in memory. "
             "Only streaming over chunks lowers peak memory: a Parquet file read whole "
             "needs as much memory as CSV."},
    {"title": "Exploratory analysis: data quality",
     "bullets": [
         "Index: 15-minute step, {{json:results/eda/eda.json:index.duplicates}} duplicates, "
         "{{json:results/eda/eda.json:index.missing_steps}} missing timestamps, NaN: {{json:results/eda/eda.json:missing_zeros.nan_values}}",
         "Gaps are coded as zeros: the client is not connected yet. "
         "{{json:results/eda/eda.json:missing_zeros.clients_by_first_year.2011}} clients have data from 2011, another "
         "{{json:results/eda/eda.json:missing_zeros.clients_by_first_year.2012}} from 2012",
         "Models use {{json:results/eda/eda.json:missing_zeros.clients_active}} clients with data from 2012-01-01 to the end",
         "DST change: in March the 01:00 hour is zero for everyone, in October it holds two hours; fixed",
         "Outliers (robust z > 5): {{json:results/eda/eda.json:outliers.flagged_share_pct}} % of client-days, "
         "most of them on 25 December and 1 January",
     ],
     "image": FIG + "01_daily_total.png"},
    {"title": "Seasonality and stationarity",
     "bullets": [
         "Daily seasonal strength (STL, hourly, 2014): {{json:results/eda/eda.json:stl_hourly_2014.seasonal_strength}}; "
         "weekly (daily series): {{json:results/eda/eda.json:stl_daily.seasonal_strength}}, weekends hardly differ from weekdays",
         "Daily series: ADF p = {{json:results/eda/eda.json:stationarity.0.adf_p}} (non-stationary), "
         "after the 7-day difference ADF p < 0.001 and KPSS p > 0.05",
         "ACF of the hourly series: peaks at 24, 48, … h → lags 24, 48, 72, 168, 336 h as features",
     ],
     "image": FIG + "03_acf_pacf.png"},
    {"title": "External factors: temperature and holidays",
     "bullets": [
         "Correlation of daily load with temperature: Pearson {{json:results/eda/eda.json:external.pearson_temp|.2f}}, "
         "Spearman {{json:results/eda/eda.json:external.spearman_temp|.2f}}: load is higher in summer (cooling)",
         "Below 15 °C the relation almost disappears: electric heating is not visible for these clients",
         "Holiday on a weekday: −{{json:results/eda/eda.json:external.holiday_drop_pct|.1f}} % vs a normal weekday",
     ],
     "image": FIG + "05_temp_vs_load.png"},
    {"title": "Typical profiles: k-means",
     "bullets": [
         "48 features per client: mean weekday and weekend hour divided by the client's mean",
         "k = 2…10; chosen k = {{json:results/clusters/result.json:k}}, the best silhouette ({{json:results/clusters/result.json:silhouette}}) "
         "among k ≥ 3 without clusters of fewer than 5 clients",
     ],
     "image": FIG + "06_cluster_profiles.png"},
    {"title": "Forecast setup",
     "bullets": [
         "Every test day at midnight: a forecast for 24 hours using only data before midnight",
         "Baselines: naive (last value), seasonal naive 24 h and 168 h",
         "ETS: 24 daily series (one per hour), weekly season, refitted every day, window of "
         "{{json:results/forecast/params.json:ets_window_days}} days (chosen on September 2014)",
         "Ridge regression and gradient boosting: lags 24–336 h, daily and weekly means, hour, day of week, month, holiday, temperature",
         "Regression and boosting hyperparameters: time-based cross-validation (TimeSeriesSplit, 4 folds of 28 days), no shuffling",
         "Metrics: MAE, RMSE, MAPE, sMAPE, MASE (scale: MAE of seasonal naive 24 h on training data)",
     ]},
    {"title": "Results: accuracy and time",
     "table": "results/forecast/summary.csv|model,total_mae,total_rmse,total_mape,total_mase,mase_mean,smape_mean,fit_s_mean,wins_mase",
     "note": "\"Total\" is the load of all clients together, kWh per hour; \"client\" is the mean over {{json:results/forecast/params.json:n_clients}} clients; "
             "time is training plus 92 forecasts on one client series; \"best for clients\" is the number of clients where the model had the lowest MASE."},
    {"title": "Error analysis",
     "bullets": [
         "December is the hardest month: holidays and the days before them; the worst day is 25 December",
         "Boosting sMAPE: {{cell:results/forecast/errors_by_day_type.csv:day_type=holiday:gbm|.1f}} % on holidays vs "
         "{{cell:results/forecast/errors_by_day_type.csv:day_type=working day:gbm|.1f}} % on working days",
         "Easiest cluster: weekday daytime profile (median boosting MASE "
         "{{cell:results/forecast/errors_mase_by_cluster.csv:cluster=3:gbm|.2f}})",
     ],
     "image": FIG + "08_errors.png"},
    {"title": "SQLite: tables and queries",
     "bullets": [
         "6 tables: clients, clusters, calendar, daily load ({{json:results/sql/db.json:rows.load_daily|,.0f}} rows), "
         "models, forecast metrics",
         "11 queries: JOIN, GROUP BY/HAVING, CTE, subqueries, RANK, ROW_NUMBER, LAG, moving average",
         "Aggregation by cluster and month: pandas {{json:results/report/ratios.json:pandas_seconds|.2f}} s, "
         "SQLite {{json:results/report/ratios.json:sqlite_seconds|.2f}} s",
         "Peak memory: pandas {{json:results/report/ratios.json:pandas_peak_rss_mb|.0f}} MB, "
         "SQLite {{json:results/report/ratios.json:sqlite_peak_rss_mb|.0f}} MB, it reads only what it needs from disk",
     ]},
    {"title": "SQLite: ER diagram", "full": FIG + "09_er.png"},
    {"title": "Streamlit dashboard", "full": "report/dashboard.png", "bare": True,
     "caption": "Streamlit dashboard: filters (period, cluster or client, series, models) and four tabs, here the forecast"},
    {"title": "Conclusions",
     "bullets": [
         "Boosting on lags is the best: total MAE {{cell:results/forecast/summary.csv:model=gbm:total_mae|,.0f}} kWh vs "
         "{{cell:results/forecast/summary.csv:model=snaive24:total_mae|,.0f}} for seasonal naive 24 h; best for "
         "{{cell:results/forecast/summary.csv:model=gbm:wins_mase}} of {{json:results/forecast/params.json:n_clients}} clients",
         "Ridge regression is almost as accurate and trains in {{cell:results/forecast/summary.csv:model=ridge:fit_s_mean|.3f}} s per series",
         "ETS on the hourly series lost to seasonal naive; ETS on 24 daily series beat it but is the slowest",
         "Four profiles; the evening-peak profile is the largest, the weekday daytime one is the easiest to forecast",
         "Limitations: actual rather than forecast temperature; few holidays in the test; one weather point for the whole country",
     ]},
    {"title": "Summary",
     "stats": [
         ("−{{json:results/report/ratios.json:gbm_vs_snaive24_pct|.0f}} %", "total MAE of boosting vs seasonal naive 24 h"),
         ("{{cell:results/forecast/summary.csv:model=gbm:wins_mase}} of {{json:results/forecast/params.json:n_clients}}",
          "clients for which boosting is the best model"),
         ("{{json:results/clusters/result.json:k}}", "typical load profiles (k-means)"),
         ("{{json:results/ingest.json:convert.csv_mb|.0f}} → {{json:results/ingest.json:convert.parquet_mb|.0f}} MB",
          "CSV → Parquet, read {{json:results/report/ratios.json:csv_over_parquet_time|.0f}} times faster"),
     ],
     "footer": "Thank you"},
]


def add_text(slide, left, top, width, height, lines, size=16):
    box = slide.shapes.add_textbox(left, top, width, height).text_frame
    box.word_wrap = True
    for i, line in enumerate(lines):
        p = box.paragraphs[0] if i == 0 else box.add_paragraph()
        p.text = ("• " if len(lines) > 1 else "") + resolve(line)
        p.font.size = Pt(size)
        p.space_after = Pt(6)


def text_height(lines, size, width) -> int:
    """Rough height of a bullet block: ~0.5 em per character, 1.2 em per line plus paragraph spacing."""
    per_line = int(width / Pt(size * 0.5))
    rows = sum(-(-len(resolve(line)) // per_line) for line in lines)
    return int(rows * Pt(size * 1.2) + len(lines) * Pt(6))


def add_full_picture(slide, path: Path, top, max_height, width=Cm(31.8)) -> None:
    """Picture as large as fits under the title, centred."""
    pic = slide.shapes.add_picture(str(path), Cm(1), top, width=width)
    if pic.height > max_height:
        pic.width, pic.height = int(pic.width * max_height / pic.height), max_height
    pic.left = (Cm(33.867) - pic.width) // 2


def wide(path: Path) -> bool:
    if not path.exists():
        return False
    with Image.open(path) as im:
        return im.width >= 2 * im.height


def add_stats(slide, stats, top) -> None:
    """Big number over a short label, two per row."""
    for i, (value, label) in enumerate(stats):
        left, row_top = Cm(1 + 16.2 * (i % 2)), top + Cm(6.2) * (i // 2)
        add_text(slide, left, row_top, Cm(15.5), Cm(2.6), [value], size=44)
        slide.shapes[-1].text_frame.paragraphs[0].font.bold = True
        slide.shapes[-1].text_frame.paragraphs[0].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        add_text(slide, left, row_top + Cm(2.6), Cm(15.5), Cm(2.4), [label], size=20)


def add_table(slide, spec: str, top) -> None:
    rows = csv_table(spec)
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Cm(1), top, Cm(31.8), Cm(0.8) * len(rows))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = shape.table.cell(i, j)
            cell.text = value
            cell.text_frame.paragraphs[0].font.size = Pt(13 if len(row) > 6 else 16)


def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Cm(33.867), Cm(19.05)
    blank = prs.slide_layouts[6]
    for n, spec in enumerate(SLIDES, 1):
        slide = prs.slides.add_slide(blank)
        if "subtitle" in spec:
            add_text(slide, Cm(2), Cm(6), Cm(30), Cm(3), [spec["title"]], size=36)
            add_text(slide, Cm(2), Cm(10), Cm(30), Cm(2), [spec["subtitle"]], size=18)
            continue
        if not spec.get("bare"):
            title = slide.shapes.add_textbox(Cm(1), Cm(0.6), Cm(31), Cm(2)).text_frame
            title.text = spec["title"]
            title.paragraphs[0].font.size = Pt(32)
            title.paragraphs[0].font.bold = True
            title.paragraphs[0].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        top = Cm(3)
        if spec.get("bare"):  # picture over the whole slide, the caption names it
            add_full_picture(slide, ROOT / spec["full"], Cm(0.3), Cm(17), Cm(33.2))
            add_text(slide, Cm(1), Cm(17.4), Cm(31), Cm(1), [spec["caption"]], size=16)
            slide.shapes[-1].text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        elif "table" in spec:
            add_table(slide, spec["table"], top)
            add_text(slide, Cm(1), Cm(14.5), Cm(31.8), Cm(4), [spec["note"]], size=16)
        elif "full" in spec:
            caption = spec.get("caption")
            add_full_picture(slide, ROOT / spec["full"], Cm(2.5), Cm(14.6 if caption else 15.9), Cm(32.8))
            if caption:
                add_text(slide, Cm(1), Cm(17.2), Cm(31.8), Cm(1.2), [caption], size=15)
                slide.shapes[-1].text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        elif "stats" in spec:
            add_stats(slide, spec["stats"], Cm(3.2))
            add_text(slide, Cm(1), Cm(15.8), Cm(31.8), Cm(1.5), [spec["footer"]], size=24)
            slide.shapes[-1].text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        elif "image" in spec and wide(ROOT / spec["image"]):  # wide figure: text on top, picture below at full width
            height = text_height(spec["bullets"], 17, Cm(31.8))
            add_text(slide, Cm(1), top, Cm(31.8), height, spec["bullets"], size=17)
            pic_top = top + height + Cm(0.4)
            add_full_picture(slide, ROOT / spec["image"], pic_top, Cm(18.3) - pic_top, Cm(32.8))
        elif "image" in spec and (ROOT / spec["image"]).exists():
            add_text(slide, Cm(1), top, Cm(12.5), Cm(15.5), spec["bullets"], size=17)
            slide.shapes.add_picture(str(ROOT / spec["image"]), Cm(14), top, width=Cm(19.3))
        else:
            add_text(slide, Cm(1), top, Cm(31.8), Cm(15.5), spec["bullets"], size=22)
        num = slide.shapes.add_textbox(Cm(32), Cm(18), Cm(1.5), Cm(0.8)).text_frame
        num.text = str(n)
        num.paragraphs[0].font.size = Pt(10)
    out = ROOT / "report" / "slides.pptx"
    prs.save(out)
    return out


if __name__ == "__main__":
    derive()
    print(f"saved {build()}")
