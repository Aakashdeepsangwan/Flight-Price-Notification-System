"""Normalizes raw Travelpayouts API payloads into the internal price-snapshot schema.

Internal snapshot schema (one dict per row -> maps 1:1 to the TimescaleDB
hypertable described in `src/storage/schema.sql`):

    origin          str            IATA code of the departure city
    destination     str            IATA code of the destination city
    departure_at    str | None     ISO 8601 departure timestamp
    return_at       str | None     ISO 8601 return timestamp (None for one-way)
    airline         str | None     IATA airline code
    flight_number   int | None
    price           float
    currency        str | None
    stops           int | None     number of layovers (0 = non-stop)
    expires_at      str | None     when the quoted price is no longer reliable
    fetched_at      str            ISO 8601 timestamp of when *we* polled the API
    source          str            always "travelpayouts" for now

`snapshots_to_dataframe()` additionally attaches `origin_city` /
`destination_city` (full city names, e.g. "DEL" -> "New Delhi") by joining
against Travelpayouts' own reference data — see `reference.py`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.reference import get_city_names

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "origin",
    "destination",
    "departure_at",
    "return_at",
    "airline",
    "flight_number",
    "price",
    "currency",
    "stops",
    "expires_at",
    "fetched_at",
    "source",
]

DISPLAY_COLUMNS = [
    "origin",
    "origin_city",
    "destination",
    "destination_city",
    "departure_at",
    "return_at",
    "airline",
    "flight_number",
    "price",
    "currency",
    "stops",
    "expires_at",
    "fetched_at",
    "source",
]


def normalize_cheapest_prices(
    payload: dict[str, Any],
    origin: str,
    currency: str,
) -> list[dict[str, Any]]:
    """Flattens a `/v1/prices/cheap` response into snapshot dicts.

    Response shape: `data -> {destination: {stop_count: offer}}`, where
    `stop_count` ("0", "1", "2", ...) is the dict key.
    """
    fetched_at = _now_iso()
    snapshots: list[dict[str, Any]] = []

    for destination, offers_by_stops in (payload.get("data") or {}).items():
        for stop_count, offer in offers_by_stops.items():
            snapshots.append(
                _base_snapshot(
                    origin=origin,
                    destination=destination,
                    offer=offer,
                    currency=currency,
                    fetched_at=fetched_at,
                    stops=_safe_int(stop_count),
                )
            )
    return snapshots


def normalize_calendar_prices(
    payload: dict[str, Any],
    origin: str,
    destination: str,
    currency: str,
) -> list[dict[str, Any]]:
    """Flattens a `/v1/prices/calendar` response into snapshot dicts.

    Response shape: `data -> {yyyy-mm-dd: offer}`, one cheapest offer per
    calendar day. `offer["transfers"]` holds the stop count here (rather
    than the dict key, as in `/v1/prices/cheap`).
    """
    fetched_at = _now_iso()
    snapshots: list[dict[str, Any]] = []

    for _day, offer in (payload.get("data") or {}).items():
        snapshots.append(
            _base_snapshot(
                origin=offer.get("origin", origin),
                destination=offer.get("destination", destination),
                offer=offer,
                currency=currency,
                fetched_at=fetched_at,
                stops=offer.get("transfers"),
            )
        )
    return snapshots


def normalize_latest_prices(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flattens a `/v2/prices/latest` response into snapshot dicts.

    This endpoint isn't scoped to one route, so `price` here maps from
    the `value` field and several per-route fields (airline, flight
    number) simply aren't available.
    """
    fetched_at = _now_iso()
    snapshots: list[dict[str, Any]] = []

    for offer in payload.get("data") or []:
        snapshots.append(
            {
                "origin": offer.get("origin"),
                "destination": offer.get("destination"),
                "departure_at": offer.get("depart_date"),
                "return_at": offer.get("return_date"),
                "airline": None,
                "flight_number": None,
                "price": offer.get("value"),
                "currency": None,
                "stops": offer.get("number_of_changes"),
                "expires_at": None,
                "fetched_at": fetched_at,
                "source": "travelpayouts",
            }
        )
    return snapshots


def snapshots_to_dataframe(
    snapshots: list[dict[str, Any]],
    add_city_names: bool = True,
) -> pd.DataFrame:
    """Converts a list of flat snapshot dicts into a typed pandas DataFrame.

    This is the step that turns the "list of dicts, one dict per row"
    output of the `normalize_*` functions above into the shape every
    downstream ML step (feature engineering, walk-forward CV, model
    training) actually expects: a typed, tabular `DataFrame`.

    Column dtypes:
        departure_at / return_at / expires_at / fetched_at -> datetime64[ns, UTC]
        price                                               -> float64
        stops / flight_number                               -> nullable Int64
        origin / destination / airline / currency / source  -> category

    If `add_city_names` is True (default), also attaches `origin_city` /
    `destination_city` full-name columns via Travelpayouts' cached city
    reference data (`reference.get_city_names()`). Falls back to blank
    city columns (with a logged warning) if the reference lookup can't be
    fetched or cached, e.g. no network access — this never blocks the
    core price data from being returned.
    """
    df = pd.DataFrame(snapshots, columns=CSV_COLUMNS)

    for col in ("departure_at", "return_at", "expires_at", "fetched_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["stops"] = pd.to_numeric(df["stops"], errors="coerce").astype("Int64")
    df["flight_number"] = pd.to_numeric(df["flight_number"], errors="coerce").astype("Int64")

    for col in ("origin", "destination", "airline", "currency", "source"):
        df[col] = df[col].astype("category")

    if add_city_names:
        try:
            city_names = get_city_names()
            df["origin_city"] = df["origin"].astype(str).map(city_names)
            df["destination_city"] = df["destination"].astype(str).map(city_names)
        except Exception as exc:  # noqa: BLE001 - reference lookup must never break the core pipeline
            logger.warning("Could not attach city names (%s); leaving origin_city/destination_city blank", exc)
            df["origin_city"] = pd.NA
            df["destination_city"] = pd.NA
    else:
        df["origin_city"] = pd.NA
        df["destination_city"] = pd.NA

    return df[DISPLAY_COLUMNS]


def write_snapshots_csv(
    snapshots: list[dict[str, Any]],
    path: str | Path,
    append: bool = True,
) -> Path:
    """Writes snapshots to a CSV file — the final, ML-ready form.

    Appends to `path` by default so repeated fetch runs accumulate into
    one growing price-history file (rather than each run overwriting the
    last), which is what Phase 1-3 train against. Pass `append=False` to
    overwrite instead.
    """
    path = Path(path)
    if not snapshots:
        return path

    df = snapshots_to_dataframe(snapshots)

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    write_header = not (append and file_exists)
    mode = "a" if append and file_exists else "w"

    df.to_csv(path, mode=mode, header=write_header, index=False)
    return path


def _base_snapshot(
    *,
    origin: str,
    destination: str,
    offer: dict[str, Any],
    currency: str,
    fetched_at: str,
    stops: int | None,
) -> dict[str, Any]:
    return {
        "origin": origin.upper(),
        "destination": destination.upper(),
        "departure_at": offer.get("departure_at"),
        "return_at": offer.get("return_at"),
        "airline": offer.get("airline"),
        "flight_number": offer.get("flight_number"),
        "price": offer.get("price"),
        "currency": currency.upper(),
        "stops": stops,
        "expires_at": offer.get("expires_at"),
        "fetched_at": fetched_at,
        "source": "travelpayouts",
    }


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
