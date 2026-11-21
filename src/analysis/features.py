"""Builds the feature table that every ML phase reads, in three tasks.

    Input   load_snapshots()           every fare from every collection round, as stored
    Task 1  clean_and_encode()         parse types, check rules, one row per trip per round,
                                       encode the text columns
    Task 2  add_rolling_features()     past-only, per route: 30-day average, 30-day min,
                                       7-day std; each row looks only at rounds before its own
    Task 3  add_stateless_features()   computed from the row alone

    build_features() runs Input -> Task 1 -> Task 2 -> Task 3.

    model_matrix(features) then turns that feature table into the model table: only the
    columns a model learns from, as numbers, with no empty cells.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.normalizer import _local_hour, canonicalize_snapshot, drop_redundant_offers

logger = logging.getLogger(__name__)

DEFAULT_RUNS_LOG = Path("data/logs/runs.jsonl")
DEFAULT_ROUTES_CONFIG = Path("config/routes.json")

TRIP_KEY = ["origin", "destination", "trip_type", "departure_date", "return_date"]
ROUTE_KEY = ["origin", "destination", "trip_type"]

# Fare columns kept by Task 1, next to TRIP_KEY and round_at.
FARE_COLUMNS = [
    "price", "airline", "flight_number", "origin_airport", "destination_airport",
    "stops", "return_stops", "max_stops", "flight_duration_min", "departure_hour", "return_hour",
    "departure_at", "return_at", "fetched_at", "found_at",
]
WHOLE_NUMBER_COLUMNS = [
    "flight_number", "stops", "return_stops", "max_stops", "flight_duration_min",
    "departure_hour", "return_hour",
]

# Columns in the snapshots that Task 1 does not keep, and why.
DROPPED_COLUMNS = {
    "origin_city": "full name of `origin` (CSV only): same information",
    "destination_city": "full name of `destination` (CSV only): same information",
    "expires_at": "always empty from this endpoint",
    "source": "always travelpayouts",
    "source_endpoint": "always prices_for_dates (checked by a rule first)",
    "currency": "always USD (checked by a rule first)",
}
EXPECTED_CURRENCY = "USD"            # config/fetch.json; prices in another currency aren't comparable
EXPECTED_ENDPOINT = "prices_for_dates"  # stops, return_stops and duration are defined by this endpoint

# Text columns encoded as categories, and the smallest share of rows a value needs to keep
# its own category; rarer values become "other". Airports get a lower bar than airlines
# because the small airports price very differently (London City +56%, Luton -15%).
CATEGORY_MIN_SHARE = {"airline": 0.005, "origin_airport": 0.001, "destination_airport": 0.001}
OTHER = "other"

# Codes for regional partners that fly under the main airline's flight numbers.
AIRLINE_ALIASES = {
    "QK": "AC",  # Jazz = Air Canada Express (AC 7952 and QK 7952 are the same flight)
    "MQ": "AA",  # Envoy = American Eagle
    "OH": "AA",  # PSA = American Eagle
}

# (start, end) as "MM-DD"; a window may cross New Year.
HOLIDAY_WINDOWS = [("12-18", "01-04")]
DEPARTURE_WEEKEND = [4, 5, 6]  # Fri, Sat, Sun: one-way departures +3.9%
RETURN_WEEKEND = [5, 6]        # Sat, Sun: +6.9% / +8.3% vs Wednesday; Friday returns were not pricier

# Model table (model_matrix). Only these feature-table columns go to a model; names, dates,
# raw prices, dollar amounts and text stay out (they name the row, or a feature already
# carries their information).
MODEL_CATEGORIES = ["destination_cat", "airline_cat", "origin_airport_cat", "destination_airport_cat"]
MODEL_NUMBERS = [
    "is_round_trip", "stops", "return_stops", "max_stops", "flight_duration_min", "is_direct", "stay_days",
    "price_vs_route_avg_30d", "price_vs_route_min_30d", "route_std_7d",
    "days_until_departure", "price_age_days",
    "departure_is_weekend", "return_is_weekend", "is_holiday_season",
]
MODEL_CALENDAR = ["departure_hour", "return_hour", "departure_dow", "return_dow", "departure_month"]

# Empty cells a model can't take, and what they become.
# "Doesn't apply": a one-way trip has no return; is_round_trip = 0 tells the model why.
ONE_WAY_FILL = {
    "return_stops": 0, "stay_days": 0, "return_is_weekend": 0,
    "return_hour": -1, "return_dow": -1,  # -1, because 0 already means midnight / Monday
}
# "No history yet": the route's first round. The price is treated as the route's usual one.
NO_HISTORY_FILL = {"price_vs_route_avg_30d": 1.0, "price_vs_route_min_30d": 1.0, "route_std_7d": 0.0}

# For linear models: hours as parts of the day (0-5, 6-11, 12-17, 18-23).
HOUR_BAND_EDGES = [0, 6, 12, 18, 24]
HOUR_BANDS = ["night", "morning", "afternoon", "evening"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def build_features(runs_log: Path = DEFAULT_RUNS_LOG, root: Path | None = None) -> pd.DataFrame:
    """Input -> Task 1 -> Task 2 -> Task 3: one row per (trip, round) with every feature."""
    df = clean_and_encode(load_snapshots(runs_log, root))
    df = add_rolling_features(df)
    df = add_stateless_features(df)
    logger.info(
        "Built %d rows: %d rounds, %d trips, %d routes",
        len(df), df["round_at"].nunique(), df.groupby(TRIP_KEY).ngroups, df.groupby(ROUTE_KEY).ngroups,
    )
    return df


# ----------------------------------------------------------------------------- Input


def load_snapshots(runs_log: Path = DEFAULT_RUNS_LOG, root: Path | None = None) -> pd.DataFrame:
    """Every fare from every round listed in `runs_log`, as stored (Task 1 cleans it).

    Adds `round_at`, the round's `started_at`. Rounds whose snapshot file is missing
    are skipped with a warning.
    """
    root = root if root is not None else runs_log.parent.parent.parent
    frames = []
    for line in runs_log.read_text().splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        path = root / run["snapshot_file"]
        if not path.exists():
            logger.warning("Skipping round %s: %s not found", run.get("started_at"), path)
            continue
        rows = drop_redundant_offers([canonicalize_snapshot(r) for r in json.loads(path.read_text())])
        if rows:
            frames.append(pd.DataFrame(rows).assign(round_at=run["started_at"]))
    if not frames:
        raise ValueError(f"No snapshot rounds could be loaded from {runs_log}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------- Task 1: cleaning + encoding


def clean_and_encode(
    raw: pd.DataFrame,
    categories: dict[str, list[str]] | None = None,
    min_share: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Task 1: one clean, typed row per (trip, round), with the text columns encoded.

    Cleaning
        keep columns   TRIP_KEY, round_at and FARE_COLUMNS; DROPPED_COLUMNS lists the rest
                       and why
        parse types    round_at, fetched_at, departure_at, return_at -> UTC datetimes;
                       found_at -> its day; price -> number; stops, durations, flight numbers
                       -> whole numbers. return_hour (local hour of the return flight) is read
                       from the raw text before the UTC conversion, like departure_hour.
                       departure_date / return_date stay "yyyy-mm-dd" text because they name
                       the trip (one-way trips get "" so they stay groupable); Task 3 turns
                       them into dates for date maths.
        fill gaps      max_stops of a one-way trip = stops (its only direction). This is the
                       true value, not a guess. Gaps that mean "doesn't apply" (return_stops
                       of a one-way trip) stay empty; each model fills those its own way.
        check rules    drop rows that break a rule, with a warning. The API data has passed
                       every rule so far; the checks are there in case that changes.
        one per trip   several fares for one trip in one round -> keep the cheapest
                       (what a buyer would see)

    Encoding (one-hot, where a model needs it, happens inside that model)
        is_round_trip            1 or 0
        destination_cat          category; the list comes from config/routes.json, so the
                                 codes don't shift as data arrives
        airline_cat              category; regional partners merged (QK -> AC)
        origin_airport_cat,      category; a city code covers several airports (YTO = YYZ,
        destination_airport_cat  YTZ, YHM), and they price differently
        For airline and airports, values under their CATEGORY_MIN_SHARE of rows (override
        with `min_share`) become "other". `origin` is not encoded: every route starts in YTO,
        so it carries no information yet.

    When scoring new rounds with a trained model, pass `categories=encoding_categories(train)`
    so the categories match the ones the model was trained on.
    """
    categories = categories or {}
    min_share = {**CATEGORY_MIN_SHARE, **(min_share or {})}
    df = raw.copy()
    for col in TRIP_KEY + FARE_COLUMNS + ["currency", "source_endpoint"]:
        if col not in df:
            df[col] = None

    # Parse types
    df["round_at"] = pd.to_datetime(df["round_at"], utc=True, format="ISO8601")
    df["fetched_at"] = _to_utc(df["fetched_at"]).fillna(df["round_at"])
    df["return_hour"] = df["return_at"].map(_local_hour)  # before UTC: 14:10 in Tokyo stays 14
    df["departure_at"] = _to_utc(df["departure_at"])
    df["return_at"] = _to_utc(df["return_at"])
    df["found_at"] = _to_utc(df["found_at"]).dt.normalize()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)
    for col in WHOLE_NUMBER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
    df["departure_date"] = df["departure_date"].fillna("").astype(str)
    df["return_date"] = df["return_date"].fillna("").astype(str)

    # Fill gaps that have a true value
    df["max_stops"] = df["max_stops"].fillna(df["stops"])

    # Check rules
    currency = df["currency"].astype("string").str.upper()
    rules = {
        "no price, or price <= 0": ~(df["price"] > 0),
        f"price not in {EXPECTED_CURRENCY}": currency.notna() & (currency != EXPECTED_CURRENCY),
        f"not from the {EXPECTED_ENDPOINT} endpoint": (
            df["source_endpoint"].notna() & (df["source_endpoint"] != EXPECTED_ENDPOINT)
        ),
        "no departure date": df["departure_date"] == "",
        "return before departure": (df["return_date"] != "") & (df["return_date"] < df["departure_date"]),
        "flight had already left when fetched": df["departure_at"] < df["fetched_at"],
    }
    broken_any = pd.Series(False, index=df.index)
    for rule, broken in rules.items():
        broken = broken.fillna(False).astype(bool)
        if broken.any():
            logger.warning("Dropping %d rows: %s", int(broken.sum()), rule)
        broken_any |= broken
    df = df[~broken_any]

    # One row per trip per round: the cheapest
    df = (
        df.sort_values("price", kind="stable")
        .drop_duplicates(TRIP_KEY + ["round_at"])
        .sort_values(TRIP_KEY + ["round_at"])
        .reset_index(drop=True)
    )

    # Encoding
    df["is_round_trip"] = (df["trip_type"] == "round_trip").astype("int8")

    destinations = categories.get("destination") or _configured_destinations(df)
    unknown = sorted(set(df["destination"]) - set(destinations))
    if unknown:
        logger.warning("Destinations missing from the category list (encoded as empty): %s", unknown)
    df["destination_cat"] = pd.Categorical(df["destination"], categories=destinations)

    df["airline_cat"] = _encode(df["airline"].replace(AIRLINE_ALIASES), categories.get("airline"), min_share["airline"])
    for col in ["origin_airport", "destination_airport"]:
        df[f"{col}_cat"] = _encode(df[col], categories.get(col), min_share[col])

    encoded = ["is_round_trip", "destination_cat", "airline_cat", "origin_airport_cat", "destination_airport_cat"]
    return df[TRIP_KEY + ["round_at"] + FARE_COLUMNS + encoded]


def encoding_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    """The category lists Task 1 used, to save with a trained model and pass back when scoring."""
    names = {"destination": "destination_cat", "airline": "airline_cat",
             "origin_airport": "origin_airport_cat", "destination_airport": "destination_airport_cat"}
    return {key: list(df[col].cat.categories) for key, col in names.items()}


def _encode(values: pd.Series, categories: list[str] | None, min_share: float) -> pd.Categorical:
    """Category column; values outside `categories` (default: those with >= `min_share` of rows) -> "other"."""
    values = values.fillna(OTHER)
    if categories is None:
        share = values.value_counts(normalize=True)
        categories = sorted(share[share >= min_share].index)
    categories = [c for c in categories if c != OTHER]
    return pd.Categorical(values.where(values.isin(categories), OTHER), categories=[*categories, OTHER])


def _to_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, format="ISO8601", errors="coerce")


def _configured_destinations(df: pd.DataFrame, config: Path = DEFAULT_ROUTES_CONFIG) -> list[str]:
    """Destination list in config/routes.json order; falls back to the data if the file can't be read."""
    try:
        return list(dict.fromkeys(route["destination"] for route in json.loads(config.read_text())))
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Could not read %s; destination categories come from the data", config)
        return sorted(df["destination"].dropna().unique())


# ------------------------------------------------------- Task 2: past-only rolling


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Task 2: three past-only features of each row's route.

    A route is every trip with the same origin, destination and trip type (YTO -> TYO round
    trip), whatever the dates. Each row looks only at the route's rounds *before* its own:
    "past 30 days" means the rounds in the 30 days before this row's round, never its own
    round or later ones. Windows are in days because rounds are not evenly spaced.

        route_avg_30d            average of every price on the route over the past 30 days
        price_vs_route_avg_30d   price / route_avg_30d; below 1 = cheaper than the route's average
        route_min_30d            lowest price on the route over the past 30 days
        price_vs_route_min_30d   price / route_min_30d; 1 or below = as cheap as the route has been
        route_std_7d             std of log price on the route over the past 7 days. Log makes
                                 it about the typical % spread, so a $150 route and a $2,000
                                 route are comparable.

    The two price_vs_ columns put the dollar values on one scale for every route.
    A route's first round has nothing to look back at, so its rows get empty values.
    """
    df = df.sort_values(ROUTE_KEY + ["round_at"], kind="stable").reset_index(drop=True)
    df["route_avg_30d"] = past_window(df, df["price"], ROUTE_KEY, "30D", "mean")
    df["price_vs_route_avg_30d"] = df["price"] / df["route_avg_30d"]
    df["route_min_30d"] = past_window(df, df["price"], ROUTE_KEY, "30D", "min")
    df["price_vs_route_min_30d"] = df["price"] / df["route_min_30d"]
    df["route_std_7d"] = past_window(df, np.log(df["price"]), ROUTE_KEY, "7D", "std")
    return df.sort_values(TRIP_KEY + ["round_at"]).reset_index(drop=True)


def past_window(df: pd.DataFrame, values: pd.Series, key: list[str], window: str, stat: str) -> np.ndarray:
    """`stat` of `values` over the `window` before each row's round, within each `key` group.

    closed="left" leaves out every row of the row's own round (and later rounds).
    `df` must be sorted by key + round_at.
    """
    series = pd.Series(np.asarray(values, dtype=float), index=df["round_at"])
    groups = [df[k].to_numpy() for k in key]
    return series.groupby(groups, sort=False).transform(
        lambda s: getattr(s.rolling(window, closed="left"), stat)()
    ).to_numpy()


# --------------------------------------------------------- Task 3: stateless (row only)


def add_stateless_features(df: pd.DataFrame, holiday_windows: list[tuple[str, str]] = HOLIDAY_WINDOWS) -> pd.DataFrame:
    """Task 3: features computed from the row alone (no other rows, no history).

    1) Timing
        days_until_departure   departure time - fetch time, in days. Both are exact UTC
                               times, so an evening flight fetched after midnight UTC is
                               never counted as already gone.
        price_age_days         fetch day - found_at (the last day an Aviasales search saw
                               this price): how old the price is. 0 = seen that same day;
                               the cache keeps prices for up to about 7 days
    2) Trip & layover structure
        stay_days              return date - departure date (empty for one-way)
        stops, return_stops, max_stops, flight_duration_min
                               kept from Task 1; flight_duration_min includes layover time
        is_direct              1 if there are no stops in either direction
    3) Calendar
        departure_dow, return_dow   0 = Monday ... 6 = Sunday
        departure_is_weekend        Fri, Sat or Sun
        return_is_weekend           Sat or Sun
        departure_month             1-12
        is_holiday_season           the departure or return falls in a `holiday_windows` window
        departure_hour, return_hour kept from Task 1 (local time of each flight)
    """
    df = df.copy()
    departure_day = pd.to_datetime(df["departure_date"], format="%Y-%m-%d")
    return_day = pd.to_datetime(df["return_date"].where(df["return_date"] != ""), format="%Y-%m-%d")

    # 1) Timing. A missing departure time falls back to noon UTC on the departure day.
    departure_time = df["departure_at"].fillna(departure_day.dt.tz_localize("UTC") + pd.Timedelta(hours=12))
    df["days_until_departure"] = (departure_time - df["fetched_at"]).dt.total_seconds() / 86400
    df["price_age_days"] = (df["fetched_at"].dt.normalize() - df["found_at"]).dt.days

    # 2) Trip & layover structure
    df["stay_days"] = (return_day - departure_day).dt.days
    most_stops = df[["stops", "return_stops"]].astype(float).max(axis=1)
    df["is_direct"] = (most_stops == 0).astype("Int64").where(most_stops.notna())

    # 3) Calendar
    df["departure_dow"] = departure_day.dt.dayofweek.astype("int8")
    df["departure_is_weekend"] = departure_day.dt.dayofweek.isin(DEPARTURE_WEEKEND).astype("int8")
    df["return_dow"] = return_day.dt.dayofweek.astype("Int64")
    df["return_is_weekend"] = return_day.dt.dayofweek.isin(RETURN_WEEKEND).astype("Int64").where(return_day.notna())
    df["departure_month"] = departure_day.dt.month.astype("int8")
    df["is_holiday_season"] = (
        _in_windows(departure_day, holiday_windows) | _in_windows(return_day, holiday_windows)
    ).astype("int8")
    return df


def _in_windows(days: pd.Series, windows: list[tuple[str, str]]) -> pd.Series:
    """True where the day's "MM-DD" falls inside any (start, end) window, in any year."""
    month_day = days.dt.strftime("%m-%d")
    inside_any = pd.Series(False, index=days.index)
    for start, end in windows:
        if start <= end:
            inside = (month_day >= start) & (month_day <= end)
        else:  # crosses New Year
            inside = (month_day >= start) | (month_day <= end)
        inside_any |= inside.fillna(False).astype(bool)
    return inside_any


# ------------------------------------------------------------------------ Model table


def model_matrix(features: pd.DataFrame, kind: str = "tree") -> pd.DataFrame:
    """The model table: only the columns a model learns from, as numbers, with no empty cells.

    It has the same rows and index as `features`, so row i of the model table is row i of
    the feature table. That's how a prediction is linked back to its trip.

    kind="tree"    For tree models (Random Forest, XGBoost). Each category becomes its code
                   (AA -> 0, AC -> 1, ...); hours, weekdays and months stay numbers.
    kind="linear"  For linear models (Ridge, Logistic Regression), which would read codes and
                   hours as amounts (UA = 15 "bigger" than AA = 0, 23:00 "bigger" than 01:00).
                   Categories, weekdays and months become one 0/1 column per value (one-hot),
                   and hours become night / morning / afternoon / evening. A one-way trip
                   gets 0 in every return column. Scale the columns (StandardScaler) inside
                   the model's pipeline.

    Empty cells are filled by ONE_WAY_FILL and NO_HISTORY_FILL. Any other empty cell is a
    data problem, so it raises an error instead of being guessed.
    """
    if kind not in ("tree", "linear"):
        raise ValueError(f'kind must be "tree" or "linear", not {kind!r}')

    values = features[MODEL_NUMBERS + MODEL_CALENDAR].astype(float)
    one_way = features["is_round_trip"] == 0
    for col, fill in ONE_WAY_FILL.items():
        values.loc[one_way, col] = values.loc[one_way, col].fillna(fill)
    values = values.fillna(NO_HISTORY_FILL)
    empty = values.isna().sum()
    if empty.any():
        raise ValueError(f"Unexpected empty cells (column: rows): {empty[empty > 0].to_dict()}")

    if kind == "tree":
        codes = pd.DataFrame(
            {col: features[col].cat.codes.astype(float) for col in MODEL_CATEGORIES}, index=features.index
        )
        return pd.concat([codes, values], axis=1)

    def one_hot(labels: pd.Series, prefix: str) -> pd.DataFrame:
        # One column per possible value, even values missing from these rows, so the
        # columns are always the same. An empty label (one-way return) gets all 0s.
        return pd.get_dummies(labels, prefix=prefix, dtype=float)

    parts = [values[MODEL_NUMBERS]]
    for col in MODEL_CATEGORIES:
        parts.append(one_hot(features[col], col.removesuffix("_cat")))
    for side in ["departure", "return"]:
        band = pd.cut(values[f"{side}_hour"], HOUR_BAND_EDGES, right=False, labels=HOUR_BANDS)
        parts.append(one_hot(band, f"{side}_time"))
        day = values[f"{side}_dow"].map(dict(enumerate(DAY_NAMES)))
        parts.append(one_hot(pd.Series(pd.Categorical(day, categories=DAY_NAMES), index=features.index), f"{side}_day"))
    month = values["departure_month"].map(dict(enumerate(MONTH_NAMES, start=1)))
    parts.append(one_hot(pd.Series(pd.Categorical(month, categories=MONTH_NAMES), index=features.index), "departure_month"))
    return pd.concat(parts, axis=1)
