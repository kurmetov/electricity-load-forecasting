"""report/report.md -> report/report.docx -> report/report.pdf (LibreOffice).

Supported markdown: #/##/### headings, paragraphs, - and 1. lists, | tables |, ![caption](path),
**bold**, `code`, and placeholders that take numbers straight from the results files, so that the
numbers in the report are the numbers of the runs:
  {{csv:path|col1,col2,...}}            a table (own line)
  {{json:path:key.sub|fmt}}             one value from a JSON file, e.g. |.1f
  {{cell:path:keycol=value:col|fmt}}    one value from a CSV row
Run: uv run --group docs python report/build.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "report.md"
sys.path.insert(0, str(ROOT))
from elf.labels import pretty  # noqa: E402


def column_digits(col: pd.Series) -> int:
    """One number of decimals for the whole column: 0 from 1000, at most 1 from 100, else at most 3."""
    vals = col.dropna().abs()
    cap = 0 if vals.empty or vals.max() >= 1000 else 1 if vals.max() >= 100 else 3
    return next(d for d in range(cap + 1) if d == cap or (vals.round(d) == vals.round(cap)).all())


def csv_table(spec: str) -> list[list[str]]:
    """'path|col1,col2:Header 2' -> rows of strings (header first), names and codes in Russian. Empty cells for NaN."""
    path, _, cols = spec.partition("|")
    df = pd.read_csv(ROOT / path.strip())
    df = df.rename(columns=lambda c: "" if c.startswith("Unnamed") else c)
    headers = [pretty(c) for c in df.columns]
    if cols.strip():
        pairs = [c.strip().split(":", 1) for c in cols.split(",")]
        df = df[[p[0] for p in pairs]]
        headers = [pretty(p[-1]) for p in pairs]
    digits = {c: column_digits(df[c]) for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    out = df.astype(object)
    for c, d in digits.items():
        out[c] = [f"{v:,.{d}f}".replace(",", "\u00a0") if pd.notna(v) else "" for v in df[c]]
    fmt = lambda v: "" if v == "" or pd.isna(v) else pretty(v)
    return [headers] + [[fmt(v) for v in row] for row in out.itertuples(index=False)]


_cache: dict[str, object] = {}


def _load(path: str):
    if path not in _cache:
        full = ROOT / path
        _cache[path] = json.loads(full.read_text()) if full.suffix == ".json" else pd.read_csv(full)
    return _cache[path]


def _format(value, fmt: str) -> str:
    if fmt:
        value = format(float(value), fmt)
    return str(value).replace(",", "\u00a0")


def resolve(text: str) -> str:
    """Replace {{json:...}} and {{cell:...}} placeholders by values from results files."""
    def json_value(m):
        obj = _load(m.group(1))
        for key in m.group(2).split("."):
            obj = obj[int(key)] if isinstance(obj, list) else obj[key]
        return _format(obj, m.group(3) or "")

    def cell_value(m):
        df = _load(m.group(1))
        key, value = m.group(2).split("=", 1)
        row = df[df[key].astype(str) == value]
        if len(row) != 1:
            raise KeyError(f"{m.group(0)}: {len(row)} rows")
        return _format(row.iloc[0][m.group(3)], m.group(4) or "")

    text = re.sub(r"\{\{json:([^:}]+):([^|}]+)(?:\|([^}]*))?\}\}", json_value, text)
    return re.sub(r"\{\{cell:([^:}]+):([^:}]+):([^|}]+)(?:\|([^}]*))?\}\}", cell_value, text)


def add_runs(par, text: str) -> None:
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if part.startswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`"):
            run = par.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(10)
        elif part:
            par.add_run(part)


def add_table(doc, rows: list[list[str]]) -> None:
    if doc.paragraphs:  # the lead-in line stays on the table's page
        doc.paragraphs[-1].paragraph_format.keep_with_next = True
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    short = len(rows) <= 12  # a short table moves to the next page whole, a long one splits with a repeated header
    for i, row in enumerate(rows):
        tr_pr = table.rows[i]._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            add_runs(cell.paragraphs[0], value)
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(9)
                run.bold = run.bold or i == 0
            cell.paragraphs[0].paragraph_format.keep_with_next = i < len(rows) - 1 if short else i == 0
    doc.add_paragraph()


def derive() -> None:
    """Small tables the report needs that are plain arithmetic on the run outputs -> results/report/."""
    out = ROOT / "results" / "report"
    out.mkdir(parents=True, exist_ok=True)
    fmt = pd.read_csv(ROOT / "results/bench/read_formats.csv").set_index("method")
    sql = pd.read_csv(ROOT / "results/bench/pandas_vs_sql.csv").set_index("method")
    summary = pd.read_csv(ROOT / "results/forecast/summary.csv").set_index("model")
    csv_t, pq_t = fmt.loc["csv, pandas default (float64)", "seconds"], fmt.loc["parquet zstd, float32", "seconds"]
    ratios = {
        "csv_over_parquet_time": csv_t / pq_t,
        "csv_over_parquet_size": fmt.loc["csv, pandas default (float64)", "file_mb"] / fmt.loc["parquet zstd, float32", "file_mb"],
        "pandas_over_sql_memory": sql["peak_rss_mb"].iloc[0] / sql["peak_rss_mb"].iloc[1],
        "pandas_seconds": sql["seconds"].iloc[0], "sqlite_seconds": sql["seconds"].iloc[1],
        "pandas_peak_rss_mb": sql["peak_rss_mb"].iloc[0], "sqlite_peak_rss_mb": sql["peak_rss_mb"].iloc[1],
        "gbm_over_ridge_time": summary.loc["gbm", "fit_s_mean"] / summary.loc["ridge", "fit_s_mean"],
        "gbm_vs_snaive24_pct": 100 * (1 - summary.loc["gbm", "total_mae"] / summary.loc["snaive24", "total_mae"]),
    }
    (out / "ratios.json").write_text(json.dumps({k: round(float(v), 3) for k, v in ratios.items()}, indent=2))
    mem = pd.read_csv(ROOT / "results/forecast/memory_profile.csv").set_index("model")
    tm = summary[["total_fit_s", "fit_s_mean"]].join(mem["peak_mb"]).rename(
        columns={"total_fit_s": "fit_s_total_series", "fit_s_mean": "fit_s_mean_clients", "peak_mb": "peak_tracemalloc_mb"})
    tm.reset_index().to_csv(out / "time_memory.csv", index=False)
    # last successful run of every experiment = the run whose files the report reads
    runs = [json.loads(l) for l in (ROOT / "runs/registry.jsonl").read_text().splitlines() if l.strip()]
    last = {}
    for r in runs:
        if r["status"] == "ok":
            last[r["experiment"]] = r
    order = ["ingest", "eda", "cluster", "forecast", "sql", "bench"]
    pd.DataFrame([{"step": last[e]["tags"].get("step"), "run_id": last[e]["run_id"], "commit": last[e]["git_commit"][:7],
                   "seconds": last[e]["elapsed_s"]} for e in order if e in last]).to_csv(out / "runs.csv", index=False)


def build(src: Path = SRC) -> Path:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    for section in doc.sections:
        section.left_margin = section.right_margin = Cm(2)
    lines = [l if l.startswith("{{csv:") else resolve(l) for l in src.read_text(encoding="utf-8").splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            doc.add_heading(line[level:].strip(), level=0 if level == 1 else level - 1)
        elif m := re.match(r"!\[(.*)\]\((.*)\)", line):
            doc.add_picture(str(ROOT / m.group(2)), width=Cm(16.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(m.group(1))
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].italic = True
        elif m := re.match(r"\{\{csv:(.*)\}\}", line):
            add_table(doc, csv_table(m.group(1)))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            add_table(doc, rows)
            continue
        elif re.match(r"(- |\d+\. )", line):
            numbered = not line.startswith("- ")
            add_runs(doc.add_paragraph(style="List Number" if numbered else "List Bullet"), re.sub(r"^(- |\d+\. )", "", line))
        else:
            par = []
            while i < len(lines) and lines[i].strip() and not re.match(r"(#|!\[|\{\{|\||- |\d+\. )", lines[i]):
                par.append(lines[i].strip())
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Cm(1)
            add_runs(p, " ".join(par))
            continue
        i += 1
    out = src.with_suffix(".docx")
    doc.save(out)
    return out


if __name__ == "__main__":
    derive()
    docx_path = build()
    print(f"saved {docx_path}")
    if "--no-pdf" not in sys.argv:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(docx_path.parent), str(docx_path)],
                       check=True, capture_output=True)
        print(f"saved {docx_path.with_suffix('.pdf')}")
