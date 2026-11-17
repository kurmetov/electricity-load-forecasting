"""SQLite: load the derived tables, run the queries from sql/queries.sql into pandas, draw the ER diagram."""

from __future__ import annotations

import sqlite3
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from elf import config as cfg
from elf import data, external

OUT = cfg.RESULTS / "sql"
MODEL_INFO = {
    "naive": ("baseline", "last observed value, held for 24 h"),
    "snaive24": ("baseline", "same hour of the previous day"),
    "snaive168": ("baseline", "same hour one week ago"),
    "ets": ("model", "ETS on 24 daily series (one per hour), weekly season, refit daily"),
    "ridge": ("model", "ridge regression on lags 24..336 h, calendar, temperature"),
    "gbm": ("model", "HistGradientBoosting on the same features"),
}


def connect() -> sqlite3.Connection:
    cfg.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(cfg.DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def build_db(chunksize: int = 50_000) -> dict:
    """Create the schema and fill it. load_daily is written in chunks of `chunksize` rows."""
    t0 = time.perf_counter()
    con = connect()
    con.executescript((cfg.SQL_DIR / "schema.sql").read_text())

    clusters = pd.read_csv(cfg.RESULTS / "clusters" / "summary.csv")
    clusters = clusters.rename(columns={"cluster": "cluster_id"})[
        ["cluster_id", "name", "peak_hour_weekday", "night_to_day", "weekend_to_weekday", "n_clients"]]
    clusters.to_sql("clusters", con, if_exists="append", index=False)

    clients = data.load_clients().merge(
        pd.read_csv(cfg.RESULTS / "clusters" / "clusters.csv")[["client_id", "cluster"]], on="client_id", how="left")
    clients = clients.rename(columns={"cluster": "cluster_id"})
    for col in ("first_day", "last_day"):
        clients[col] = clients[col].dt.strftime("%Y-%m-%d")
    clients["active"] = clients["active"].astype(int)
    clients["cluster_id"] = clients["cluster_id"].astype("Int64")
    clients[["client_id", "first_day", "last_day", "zero_days_share", "mean_kwh_day", "active", "cluster_id"]] \
        .to_sql("clients", con, if_exists="append", index=False)

    cal = external.load_calendar().reset_index()
    cal["date"] = cal["date"].dt.strftime("%Y-%m-%d")
    cal[["is_weekend", "is_holiday"]] = cal[["is_weekend", "is_holiday"]].astype(int)
    cal.to_sql("calendar", con, if_exists="append", index=False)

    daily = data.load_daily()
    long = daily.rename_axis("date").reset_index().melt(id_vars="date", var_name="client_id", value_name="kwh")
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")
    long[["client_id", "date", "kwh"]].to_sql("load_daily", con, if_exists="append", index=False, chunksize=chunksize)

    pd.DataFrame([(m, k, d) for m, (k, d) in MODEL_INFO.items()], columns=["model", "kind", "description"]) \
        .to_sql("models", con, if_exists="append", index=False)
    metrics = pd.read_csv(cfg.RESULTS / "forecast" / "metrics.csv")
    metrics.to_sql("forecast_metrics", con, if_exists="append", index=False)
    con.commit()
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("clusters", "clients", "calendar", "load_daily", "models", "forecast_metrics")}
    con.close()
    return {"rows": counts, "seconds": round(time.perf_counter() - t0, 1),
            "db_mb": round(cfg.DB_PATH.stat().st_size / 2**20, 1)}


def read_queries() -> dict[str, str]:
    """Split sql/queries.sql on '-- name: <name>' lines."""
    queries, name, lines = {}, None, []
    for line in (cfg.SQL_DIR / "queries.sql").read_text().splitlines():
        if line.startswith("-- name:"):
            if name:
                queries[name] = "\n".join(lines).strip()
            name, lines = line.split(":", 1)[1].strip(), []
        elif name:
            lines.append(line)
    if name:
        queries[name] = "\n".join(lines).strip()
    return queries


def run_queries() -> pd.DataFrame:
    """Run every query, save each result to results/sql/<name>.csv, return timings."""
    OUT.mkdir(parents=True, exist_ok=True)
    con = connect()
    rows = []
    for name, sql in read_queries().items():
        t0 = time.perf_counter()
        try:
            df = pd.read_sql_query(sql, con)
        except (sqlite3.OperationalError, pd.errors.DatabaseError) as exc:
            print(f"{name}: failed: {exc}")
            rows.append({"query": name, "rows": -1, "seconds": None, "error": str(exc)})
            continue
        df.to_csv(OUT / f"{name}.csv", index=False)
        rows.append({"query": name, "rows": len(df), "seconds": round(time.perf_counter() - t0, 3), "error": ""})
    con.close()
    timing = pd.DataFrame(rows)
    timing.to_csv(OUT / "timing.csv", index=False)
    return timing


def draw_er(path=None) -> None:
    """ER diagram from the live database: tables as boxes, foreign keys as lines."""
    con = connect()
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    cols = {t: con.execute(f"PRAGMA table_info({t})").fetchall() for t in tables}
    fks = {t: con.execute(f"PRAGMA foreign_key_list({t})").fetchall() for t in tables}
    con.close()
    pos = {"clusters": (0, 1), "clients": (1, 1), "load_daily": (2, 1), "calendar": (3, 1),
           "models": (2, 0), "forecast_metrics": (1, 0)}
    fig, ax = plt.subplots(figsize=(12, 6))
    box_w, line_h = 0.85, 0.045
    row_y, box_x = {}, {}
    for t in tables:
        x, y = pos.get(t, (3, 0))
        top = 0.45 + y * 0.55
        h = line_h * (len(cols[t]) + 1)
        ax.add_patch(plt.Rectangle((x, top - h), box_w, h, fill=False, lw=1.2))
        ax.add_patch(plt.Rectangle((x, top - line_h), box_w, line_h, color="#dde6f0"))
        ax.text(x + 0.02, top - line_h / 2, t, va="center", fontsize=9, weight="bold")
        fk_cols = {f[3] for f in fks[t]}
        for i, c in enumerate(cols[t]):
            mark = ("PK " if c[5] else "") + ("FK " if c[1] in fk_cols else "")
            yy = top - line_h * (i + 1.5)
            ax.text(x + 0.02, yy, f"{mark}{c[1]} : {c[2]}", va="center", fontsize=7.5)
            row_y[(t, c[1])] = yy
        box_x[t] = x
    for t, flist in fks.items():  # arrow from the FK row to the PK row, edge to edge
        for f in flist:
            ref, col, ref_col = f[2], f[3], f[4]
            right = box_x[ref] > box_x[t]
            x1 = box_x[t] + (box_w if right else 0)
            x2 = box_x[ref] + (0 if right else box_w)
            ax.annotate("", xy=(x2, row_y[(ref, ref_col)]), xytext=(x1, row_y[(t, col)]),
                        arrowprops={"arrowstyle": "-|>", "color": "C3", "lw": 1.2})
    ax.set_xlim(-0.1, 4)
    ax.set_ylim(0.0, 1.05)
    ax.axis("off")
    ax.set_title("ER diagram (arrow: foreign key -> primary key)")
    fig.tight_layout()
    fig.savefig(path or cfg.FIGURES / "09_er.png", dpi=150)
    plt.close(fig)


def run_db() -> dict:
    from elf.tracking import Run

    with Run("sql", {"db": str(cfg.DB_PATH.relative_to(cfg.ROOT)), "queries": "sql/queries.sql", "chunksize": 50_000},
             tags={"step": "S5"}) as run:
        stats = build_db()
        timing = run_queries()
        draw_er()
        for _, r in timing.iterrows():
            run.log({"rows": r["rows"], "seconds": r["seconds"]}, query=r["query"])
        data.save_json({**stats, "queries": timing.to_dict("records")}, OUT / "db.json")
        run.finish({**{f"rows.{k}": v for k, v in stats["rows"].items()}, "load_s": stats["seconds"],
                    "db_mb": stats["db_mb"], "queries": len(timing), "failed": int((timing["rows"] < 0).sum())})
    print(timing)
    return stats


if __name__ == "__main__":
    run_db()
