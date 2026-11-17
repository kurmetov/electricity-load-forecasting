"""Typical load profiles: k-means on normalised weekday + weekend hourly profiles of each client."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from elf import config as cfg
from elf import data

OUT = cfg.RESULTS / "clusters"
MIN_SIZE = 5


def client_profiles(hourly: pd.DataFrame) -> pd.DataFrame:
    """48 numbers per client: mean kWh by hour on weekdays and on weekends, divided by the client's
    overall mean, so that the shape counts and the size does not."""
    h = hourly.loc[cfg.TRAIN_START:cfg.TEST_START]
    weekend = np.where(h.index.dayofweek >= 5, "we", "wd")
    prof = h.groupby([weekend, h.index.hour]).mean()  # (2*24) x clients
    prof.index = [f"{d}_{hour:02d}" for d, hour in prof.index]
    prof = prof.T
    return prof.div(prof.mean(axis=1), axis=0)


def choose_k(x: np.ndarray, ks=range(2, 11), seed: int = 0) -> list[dict]:
    scores = []
    for k in [k for k in ks if k < len(x)]:  # silhouette needs k < number of clients
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(x)
        scores.append({"k": k, "inertia": round(float(km.inertia_), 2),
                       "silhouette": round(float(silhouette_score(x, km.labels_)), 4),
                       "min_size": int(np.bincount(km.labels_).min())})
    return scores


def pick_k(scores: list[dict], min_size: int = MIN_SIZE) -> int:
    """Highest silhouette among k >= 3 whose smallest cluster has at least `min_size` clients:
    a cluster of one or two clients is an outlier, not a typical profile."""
    ok = [s for s in scores if s["k"] >= 3 and s["min_size"] >= min_size] or [s for s in scores if s["k"] >= 3]
    return max(ok, key=lambda s: s["silhouette"])["k"]


def describe_cluster(profile: pd.Series) -> dict:
    """Peak hour, night/day ratio and weekend/weekday ratio of a cluster centre."""
    wd = profile[[c for c in profile.index if c.startswith("wd")]].to_numpy()
    we = profile[[c for c in profile.index if c.startswith("we")]].to_numpy()
    night, day = wd[0:6].mean(), wd[9:18].mean()
    return {
        "peak_hour_weekday": int(wd.argmax()),
        "night_to_day": round(float(night / day), 2),
        "weekend_to_weekday": round(float(we.mean() / wd.mean()), 2),
    }


def name_cluster(desc: dict) -> str:
    """Short name from the shape of the centre (clients are anonymous, so no sector names)."""
    if desc["night_to_day"] > 2:
        return "night"
    if desc["weekend_to_weekday"] < 0.75:
        return "weekday daytime"
    if desc["night_to_day"] > 0.75:
        return "flat around the clock"
    if desc["night_to_day"] < 0.3:
        return "day and evening, near zero at night"
    return "evening peak, moderate night"


def plot_profiles(centres: pd.DataFrame, sizes: pd.Series, names: dict) -> None:
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for cluster, row in centres.iterrows():
        label = f"{cluster}: {names[cluster]} (n={sizes[cluster]})"
        axes[0].plot(range(24), row.iloc[:24].to_numpy(), marker=".", label=label)
        axes[1].plot(range(24), row.iloc[24:].to_numpy(), marker=".")
    axes[0].set_title("Weekdays")
    axes[1].set_title("Weekends")
    for ax in axes:
        ax.set_xlabel("hour")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("share of client mean")
    fig.legend(loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Cluster centres of daily profiles (k-means)")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "06_cluster_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_clustering(k: int | None = None, seed: int = 0) -> dict:
    """Fit k-means for k = 2..10, pick k by silhouette among k >= 3 unless k is given, save labels."""
    from elf.tracking import Run

    OUT.mkdir(parents=True, exist_ok=True)
    clients = data.load_clients()
    active = clients.loc[clients["active"], "client_id"].tolist()
    with Run("cluster", {"features": "48 normalised hourly means (weekday, weekend)", "period": [cfg.TRAIN_START, cfg.TEST_START],
                         "k_given": k, "k_rule": f"max silhouette, k >= 3, smallest cluster >= {MIN_SIZE}", "seed": seed, "n_init": 20, "grid": {"k": list(range(2, 11))}},
             tags={"step": "S3"}) as run:
        prof = client_profiles(data.load_hourly(columns=active))
        x = prof.to_numpy()
        scores = choose_k(x, seed=seed)
        for s in scores:
            run.log({"inertia": s["inertia"], "silhouette": s["silhouette"], "min_size": s["min_size"]}, k=s["k"])
        pd.DataFrame(scores).to_csv(OUT / "k_selection.csv", index=False)
        if k is None:
            k = pick_k(scores)
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(x)
        # order clusters by size so that labels are stable between runs
        order = pd.Series(km.labels_).value_counts().index.tolist()
        relabel = {old: new for new, old in enumerate(order)}
        labels = pd.Series([relabel[l] for l in km.labels_], index=prof.index, name="cluster")
        centres = prof.groupby(labels).mean()
        sizes = labels.value_counts().sort_index()
        info = {c: describe_cluster(centres.loc[c]) for c in centres.index}
        names = {c: name_cluster(d) for c, d in info.items()}
        plot_profiles(centres, sizes, names)
        labels.to_frame().assign(cluster_name=labels.map(names)).rename_axis("client_id").to_csv(OUT / "clusters.csv")
        centres.to_parquet(OUT / "centres.parquet")
        summary = pd.DataFrame(info).T.assign(name=pd.Series(names), n_clients=sizes)
        summary.index.name = "cluster"
        summary.to_csv(OUT / "summary.csv")
        best = next(s for s in scores if s["k"] == k)
        result = {"k": k, "silhouette": best["silhouette"], "inertia": best["inertia"], "clients": len(labels),
                  **{f"size_{c}": int(n) for c, n in sizes.items()}}
        data.save_json(result, OUT / "result.json")
        run.finish(result)
    print(summary)
    return result


if __name__ == "__main__":
    run_clustering()
