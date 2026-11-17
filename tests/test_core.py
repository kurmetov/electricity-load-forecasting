"""Small checks of the logic that is easy to get wrong. Run: python -m pytest tests  or  python tests/test_core.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elf.clustering import pick_k
from elf.data import dst_days, fix_dst
from elf.db import read_queries
from elf.forecast import metrics, predict_baseline


def test_metrics():
    y, p = np.array([10.0, 0.0, 20.0]), np.array([12.0, 1.0, 20.0])
    m = metrics(y, p, scale=2.0)
    assert m["mae"] == 1.0 and m["mase"] == 0.5
    assert abs(m["mape"] - 10.0) < 1e-9          # zero actual is skipped: (20 % + 0 %) / 2
    assert abs(m["smape"] - 100 * (2 * 2 / 22 + 2 + 0) / 3) < 1e-9


def test_naive_uses_previous_day_only():
    idx = pd.date_range("2014-01-01", periods=72, freq="h")
    y = pd.Series(np.arange(72.0), index=idx)
    naive = predict_baseline(y, "naive")
    assert (naive["2014-01-02"] == 23.0).all()    # last hour of the day before
    assert (predict_baseline(y, "snaive24")["2014-01-03"].to_numpy() == np.arange(24.0, 48.0)).all()


def test_fix_dst():
    assert dst_days([2014]) == [pd.Timestamp("2014-03-30"), pd.Timestamp("2014-10-26")]
    idx = pd.date_range("2014-03-29", "2014-10-27", freq="h")
    h = pd.DataFrame({"a": 10.0, "b": 20.0}, index=idx)
    h.loc["2014-03-30 01:00"] = 0.0
    h.loc["2014-10-26 01:00"] *= 2
    fixed = fix_dst(h)
    assert (fixed == 10.0)["a"].all() and (fixed == 20.0)["b"].all()


def test_pick_k_skips_tiny_clusters():
    scores = [{"k": 3, "silhouette": 0.3, "min_size": 20}, {"k": 4, "silhouette": 0.4, "min_size": 20},
              {"k": 5, "silhouette": 0.5, "min_size": 1}]
    assert pick_k(scores) == 4


def test_queries_parsed():
    q = read_queries()
    bodies = ["\n".join(l for l in sql.splitlines() if not l.startswith("--")).strip() for sql in q.values()]
    assert len(q) >= 8 and all(b.upper().startswith(("SELECT", "WITH")) for b in bodies)



def test_column_digits():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "report"))
        from build import column_digits
    except ImportError:  # python-docx lives in the docs group
        return
    assert column_digits(pd.Series([33.1165, 3.99, 0.0004])) == 3
    assert column_digits(pd.Series([0.64, 11.91])) == 2
    assert column_digits(pd.Series([15.0, None, 63.0])) == 0
    assert column_digits(pd.Series([678.1, 41.6])) == 1
    assert column_digits(pd.Series([60034.67, 5948.96])) == 0

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
