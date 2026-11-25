"""Phase 1 — unsupervised models (no Buy/Wait labels needed).

    K-Means           groups routes by how their prices move (stable ... volatile)
    Isolation Forest  scores each trip's price in a round against its own
                      recent history and its route; flags unusual drops
    DBSCAN            flags trips whose whole price sequence behaves unlike
                      the rest (structural outliers, possible data problems)

Run directly:

    python -m src.ml.unsupervised
    python -m src.ml.unsupervised --k 3 --contamination 0.03
    python -m src.ml.unsupervised --out-dir models/phase1

Outputs go to `--out-dir` (default `models/phase1/`): CSV tables, fitted
models (joblib), diagnostic plots, and `summary.json`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.features import DEFAULT_RUNS_LOG, ROUTE_KEY, TRIP_KEY, build_features, past_window

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("models/phase1")

ROUTE_FEATURES = ["change_rate", "move_size", "volatility", "trend_pct_per_day", "date_spread"]
ANOMALY_FEATURES = ["log_vs_trip_7d", "log_chg_prev", "log_vs_route"]
TRIP_FEATURES = ["rel_level", "volatility", "change_rate", "total_change", "max_drop"]


# ------------------------------------------------------------------ Phase 1 inputs


def add_price_history(features: pd.DataFrame) -> pd.DataFrame:
    """Columns the Phase 1 models need, added to the feature table from build_features().

        prev_price       the trip's price in the previous round it was seen
        log_chg_prev     log(price / prev_price)
        trip_median_7d   the trip's median price over the past 7 days (its own round left out)
        log_vs_trip_7d   log(price / trip_median_7d)
        route_median     median price of the route's trips in this same round
        log_vs_route     log(price / route_median)

    The first round a trip is seen has no earlier price, so its prev_ / trip_ columns are empty.
    """
    df = features.sort_values(TRIP_KEY + ["round_at"]).reset_index(drop=True)
    trips = df.groupby(TRIP_KEY, sort=False)
    df["prev_price"] = trips["price"].shift()
    df["log_chg_prev"] = np.log(df["price"] / df["prev_price"])
    df["trip_median_7d"] = past_window(df, df["price"], TRIP_KEY, "7D", "median")
    df["log_vs_trip_7d"] = np.log(df["price"] / df["trip_median_7d"])
    df["route_median"] = df.groupby(ROUTE_KEY + ["round_at"])["price"].transform("median")
    df["log_vs_route"] = np.log(df["price"] / df["route_median"])
    return df


# --------------------------------------------------------------------------- K-Means


def route_behaviour(features: pd.DataFrame) -> pd.DataFrame:
    """Summarise how each route's prices move over time (input to K-Means).

    The columns describe movement, not price level, so long-haul and
    short-haul routes can land in the same group if they behave alike.
        change_rate         how often does the price change? (how many times the price changed / how many times we compare it with previous price)
        move_size           how big the changes are? (Eg. +10% (200→220) and −4% (250→240) => median size is 7%)
        volatility          how often and how big the changes are?
        trend_pct_per_day   average daily % change of the same trips (ignores trips coming and going)
        date_spread         date spread shows us how prices differ among dates (eg day 1, day2, day 3 - the days you fly = departure_date)
    """
    df = features.copy()
    df["prev_round_at"] = df.groupby(TRIP_KEY, sort=False)["round_at"].shift()
    steps = df.dropna(subset=["log_chg_prev"]).copy()
    steps["dt_days"] = (steps["round_at"] - steps["prev_round_at"]).dt.total_seconds() / 86400
    steps["changed"] = steps["log_chg_prev"].abs() > 1e-9
    steps["abs_move"] = steps["log_chg_prev"].abs()

    routes = df.groupby(ROUTE_KEY)
    out = pd.DataFrame({
        "n_trips": df.drop_duplicates(TRIP_KEY).groupby(ROUTE_KEY).size(),
        "median_price": routes["price"].median(),
    })
    by_step = steps.groupby(ROUTE_KEY)
    out["n_steps"] = by_step.size()
    out["change_rate"] = by_step["changed"].mean()
    out["move_size"] = np.expm1(steps[steps["changed"]].groupby(ROUTE_KEY)["abs_move"].median())
    out["move_size"] = out["move_size"].where(out["n_steps"].isna(), out["move_size"].fillna(0.0))
    out["trend_pct_per_day"] = np.expm1(by_step["log_chg_prev"].sum() / by_step["dt_days"].sum())

    log_price = np.log(df["price"]).rename("log_price")
    trip_vol = df[TRIP_KEY].join(log_price).groupby(TRIP_KEY).agg(vol=("log_price", "std"), n=("log_price", "size"))
    trip_vol = trip_vol[trip_vol["n"] >= 3].reset_index()
    out["volatility"] = trip_vol.groupby(ROUTE_KEY)["vol"].mean()  # median is 0: most trips never move

    cv = df.groupby(ROUTE_KEY + ["round_at"])["price"].agg(lambda s: s.std() / s.mean() if len(s) > 1 else np.nan)
    out["date_spread"] = cv.groupby(level=ROUTE_KEY).median()
    out["n_steps"] = out["n_steps"].fillna(0).astype(int)
    return out.reset_index()


def cluster_routes(
    routes: pd.DataFrame,
    k: int | None = None,
    k_range: range = range(2, 7),
    min_steps: int = 50,
    random_state: int = 0,
) -> tuple[pd.DataFrame, Pipeline | None, pd.DataFrame]:
    """Cluster routes on their price-movement stats.

    Routes with fewer than `min_steps` round-to-round observations are left
    unclustered (`cluster = -1`, name "too_little_data"). If `k` is None it
    is chosen by the best silhouette score in `k_range`.

    Returns (routes with cluster columns, fitted scaler+KMeans, k-selection table).
    """
    out = routes.copy()
    out["cluster"] = -1
    out["cluster_name"] = "too_little_data"
    usable = (out["n_steps"] >= min_steps) & out[ROUTE_FEATURES].notna().all(axis=1)
    X = out.loc[usable, ROUTE_FEATURES]
    if len(X) < 3:
        logger.warning("Only %d routes have enough data; skipping K-Means", len(X))
        return out, None, pd.DataFrame()

    rows = []
    Xs = StandardScaler().fit_transform(X)
    for kk in k_range:
        if kk >= len(X):
            break
        km = KMeans(n_clusters=kk, n_init=20, random_state=random_state).fit(Xs)
        rows.append({"k": kk, "inertia": km.inertia_, "silhouette": silhouette_score(Xs, km.labels_)})
    selection = pd.DataFrame(rows)
    if k is None:
        k = int(selection.loc[selection["silhouette"].idxmax(), "k"])

    model = Pipeline([
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=random_state)),
    ]).fit(X)
    out.loc[usable, "cluster"] = model.predict(X)
    out.loc[usable, "cluster_name"] = out.loc[usable, "cluster"].map(_name_clusters(out[usable]))
    return out, model, selection


def _name_clusters(clustered: pd.DataFrame) -> dict[int, str]:
    """Name clusters from calmest to most volatile by their average movement."""
    movement = clustered[["change_rate", "move_size", "volatility"]]
    movement = (movement - movement.mean()) / movement.std(ddof=0).replace(0, 1)
    order = movement.mean(axis=1).groupby(clustered["cluster"]).mean().sort_values().index.tolist()
    if len(order) == 2:
        names = ["stable", "volatile"]
    elif len(order) == 3:
        names = ["stable", "moderate", "volatile"]
    else:
        names = ["stable"] + [f"moderate_{i}" for i in range(1, len(order) - 1)] + ["volatile"]
    return dict(zip(order, names))


# ------------------------------------------------------------------ Isolation Forest


def anomaly_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest inputs. No earlier price = no evidence of a change, so 0."""
    return features[ANOMALY_FEATURES].fillna(0.0)


def fit_anomaly_detector(
    features: pd.DataFrame, contamination: float = 0.05, random_state: int = 0
) -> IsolationForest:
    """Fit on historical (trip, round) rows; `contamination` is the share expected to be unusual."""
    return IsolationForest(
        n_estimators=300, contamination=contamination, random_state=random_state
    ).fit(anomaly_matrix(features))


def score_anomalies(model: IsolationForest, features: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly columns.

        anomaly_score  lower = more unusual (IsolationForest.decision_function; < 0 means flagged)
        is_anomaly     the forest flags this row
        kind           which way the price moved: drop / rise against the trip's own
                       recent prices, else cheap_for_route / dear_for_route
        drop_alert     is_anomaly and kind == "drop" — the only flags worth alerting on
        fresh_drop     drop_alert and the price fell since the previous round; an older
                       drop stays below its 7-day median (and flagged) for up to a week
    """
    out = features.copy()
    X = anomaly_matrix(out)
    out["anomaly_score"] = model.decision_function(X)
    out["is_anomaly"] = model.predict(X) == -1
    own = out["log_vs_trip_7d"].fillna(out["log_chg_prev"]).fillna(0.0)
    out["kind"] = np.select(
        [own < 0, own > 0, out["log_vs_route"] < 0],
        ["drop", "rise", "cheap_for_route"],
        default="dear_for_route",
    )
    out["drop_alert"] = out["is_anomaly"] & (out["kind"] == "drop")
    out["fresh_drop"] = out["drop_alert"] & (out["log_chg_prev"] < 0)
    out["pct_vs_trip_7d"] = np.expm1(out["log_vs_trip_7d"])
    out["pct_vs_route"] = np.expm1(out["log_vs_route"])
    return out


# --------------------------------------------------------------------------- DBSCAN


def trip_behaviour(features: pd.DataFrame, min_rounds: int = 6) -> pd.DataFrame:
    """Summarise each trip's whole price sequence (input to DBSCAN).

    Only trips seen in at least `min_rounds` rounds are kept.
        rel_level      median log(price / route median): persistently cheap (<0) or dear (>0) for its route
        volatility     std of log price across rounds
        change_rate    share of steps where the price changed
        total_change   log(last price / first price)
        max_drop       largest single-step fall, as a positive log value
    """
    g = features.groupby(TRIP_KEY, sort=False)
    out = g.agg(
        n_rounds=("price", "size"),
        first_seen=("round_at", "min"),
        last_seen=("round_at", "max"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        rel_level=("log_vs_route", "median"),
        change_rate=("log_chg_prev", lambda s: (s.dropna().abs() > 1e-9).mean() if s.notna().any() else 0.0),
        max_drop=("log_chg_prev", lambda s: max(0.0, -s.min()) if s.notna().any() else 0.0),
    )
    out["volatility"] = g["price"].agg(lambda s: np.log(s).std())
    out["total_change"] = np.log(out["last_price"] / out["first_price"])
    out = out[out["n_rounds"] >= min_rounds]
    return out.reset_index()


def knee_eps(X: np.ndarray, min_samples: int) -> tuple[float, np.ndarray]:
    """Pick DBSCAN's eps at the knee of the sorted k-distance curve.

    The knee is the point farthest below the straight line joining the
    curve's ends. Returns (eps, sorted k-distances) so the curve can be plotted.
    """
    dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    kdist = np.sort(dist[:, -1])
    x = np.linspace(0, 1, len(kdist))
    span = kdist[-1] - kdist[0]
    y = (kdist - kdist[0]) / span if span > 0 else np.zeros_like(kdist)
    return float(kdist[int(np.argmax(x - y))]), kdist


def find_trip_outliers(
    trips: pd.DataFrame, eps: float | None = None, min_samples: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    """Run DBSCAN on scaled trip-behaviour stats; `dbscan_label == -1` marks an outlier.

    Defaults: min_samples = 2 x number of features, eps from the k-distance knee.
    Returns (trips with labels, parameters used, sorted k-distances).
    """
    out = trips.copy()
    min_samples = min_samples or 2 * len(TRIP_FEATURES)
    if len(out) <= min_samples:
        logger.warning("Only %d trips with enough rounds; skipping DBSCAN", len(out))
        out["dbscan_label"] = 0
        out["is_outlier"] = False
        return out, {"eps": None, "auto_eps": None, "min_samples": min_samples}, np.array([])
    X = StandardScaler().fit_transform(out[TRIP_FEATURES])
    auto_eps, kdist = knee_eps(X, min_samples)
    eps = eps or auto_eps
    out["dbscan_label"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    out["is_outlier"] = out["dbscan_label"] == -1
    params = {"eps": eps, "auto_eps": auto_eps, "min_samples": min_samples}
    return out, params, kdist


# ----------------------------------------------------------------------------- run


def run_phase1(
    runs_log: Path = DEFAULT_RUNS_LOG,
    out_dir: Path = DEFAULT_OUT_DIR,
    k: int | None = None,
    contamination: float = 0.05,
    eps: float | None = None,
    min_trip_rounds: int = 6,
) -> dict[str, Any]:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    features = add_price_history(build_features(runs_log))
    latest = features["round_at"].max()

    # K-Means: route behaviour groups
    routes, kmeans, k_table = cluster_routes(route_behaviour(features), k=k)
    routes.to_csv(out_dir / "route_clusters.csv", index=False)
    k_table.to_csv(out_dir / "kmeans_k_selection.csv", index=False)
    if kmeans is not None:
        joblib.dump(kmeans, out_dir / "route_kmeans.joblib")

    # Isolation Forest: fit on past rounds, score everything, report the latest round
    history = features[features["round_at"] < latest]
    forest = fit_anomaly_detector(history if len(history) else features, contamination)
    joblib.dump(forest, out_dir / "price_isoforest.joblib")
    scored = score_anomalies(forest, features)
    keep = TRIP_KEY + [
        "round_at", "price", "prev_price", "trip_median_7d", "route_median", "pct_vs_trip_7d",
        "pct_vs_route", "days_until_departure", "anomaly_score", "is_anomaly", "kind", "drop_alert",
        "fresh_drop",
    ]
    scored[keep].to_csv(out_dir / "anomaly_scores.csv", index=False)
    latest_flags = scored[(scored["round_at"] == latest) & scored["is_anomaly"]].sort_values("anomaly_score")
    latest_flags[keep].to_csv(out_dir / "anomalies_latest_round.csv", index=False)

    # DBSCAN: structurally odd trips
    trips, db_params, kdist = find_trip_outliers(trip_behaviour(features, min_trip_rounds), eps=eps)
    trips.sort_values(["is_outlier", "volatility"], ascending=False).to_csv(out_dir / "trip_outliers.csv", index=False)

    _save_plots(out_dir, k_table, kdist, db_params["eps"])

    big_drop = scored["log_vs_trip_7d"] < np.log(0.95)
    summary = {
        "data": {
            "rounds": int(features["round_at"].nunique()),
            "first_round": str(features["round_at"].min()),
            "latest_round": str(latest),
            "trip_rounds": len(features),
            "trips": int(features.groupby(TRIP_KEY).ngroups),
            "routes": int(features.groupby(ROUTE_KEY).ngroups),
        },
        "kmeans": {
            "k": int(routes.loc[routes["cluster"] >= 0, "cluster"].nunique()),
            "routes_clustered": int((routes["cluster"] >= 0).sum()),
            "cluster_sizes": routes["cluster_name"].value_counts().to_dict(),
        },
        "isolation_forest": {
            "contamination": contamination,
            "fit_rows": len(history),
            "flagged_share_all_rounds": round(float(scored["is_anomaly"].mean()), 4),
            "latest_round_flagged": int(len(latest_flags)),
            "latest_round_drop_alerts": int(latest_flags["drop_alert"].sum()),
            "latest_round_fresh_drops": int(latest_flags["fresh_drop"].sum()),
            "flag_kinds_all_rounds": scored.loc[scored["is_anomaly"], "kind"].value_counts().to_dict(),
            "drops_over_5pct_vs_7d_median": int(big_drop.sum()),
            "of_which_flagged": int((big_drop & scored["is_anomaly"]).sum()),
        },
        "dbscan": {
            **{k_: round(v, 4) if isinstance(v, float) else v for k_, v in db_params.items()},
            "trips": len(trips),
            "outliers": int(trips["is_outlier"].sum()),
            "clusters": int(trips.loc[~trips["is_outlier"], "dbscan_label"].nunique()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _save_plots(out_dir: Path, k_table: pd.DataFrame, kdist: np.ndarray, eps: float | None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping diagnostic plots")
        return

    if not k_table.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k_table["k"], k_table["inertia"], marker="o", color="#2a6fdb", label="inertia")
        ax.set(xlabel="K", ylabel="inertia", title="K-Means elbow (routes)")
        ax2 = ax.twinx()
        ax2.plot(k_table["k"], k_table["silhouette"], marker="s", color="#e07b00", label="silhouette")
        ax2.set_ylabel("silhouette")
        fig.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "kmeans_elbow.png", dpi=120)
        plt.close(fig)

    if eps is None or len(kdist) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(kdist, color="#2a6fdb")
    ax.axhline(eps, color="#e07b00", linestyle="--", label=f"eps = {eps:.3f}")
    ax.set(xlabel="trips, sorted", ylabel="distance to k-th neighbour", title="DBSCAN k-distance (trips)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dbscan_kdistance.png", dpi=120)
    plt.close(fig)


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Phase 1 unsupervised models")
    parser.add_argument("--runs-log", type=Path, default=DEFAULT_RUNS_LOG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k", type=int, default=None, help="K-Means clusters (default: best silhouette)")
    parser.add_argument("--contamination", type=float, default=0.05, help="Isolation Forest expected anomaly share")
    parser.add_argument("--eps", type=float, default=None, help="DBSCAN eps (default: k-distance knee)")
    parser.add_argument("--min-trip-rounds", type=int, default=6, help="DBSCAN: rounds a trip needs to be included")
    args = parser.parse_args()

    summary = run_phase1(args.runs_log, args.out_dir, args.k, args.contamination, args.eps, args.min_trip_rounds)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
