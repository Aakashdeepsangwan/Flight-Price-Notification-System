"""Normalizes raw Travelpayouts API payloads into the internal price-snapshot schema.

Internal snapshot schema (one dict per row -> maps 1:1 to the TimescaleDB
hypertable described in `src/storage/schema.sql`):

    origin               str            IATA code of the departure city
    destination          str            IATA code of the destination city
    departure_at         str | None     ISO 8601 departure timestamp
    return_at            str | None     ISO 8601 return timestamp (None for one-way)
    airline              str | None     IATA airline code
    flight_number        int | None
    price                float
    currency             str | None
    stops                int | None     number of layovers (0 = non-stop)
    expires_at           str | None     when the quoted price is no longer reliable
    fetched_at           str            ISO 8601 timestamp of when *we* polled the API
    source               str            always "travelpayouts" for now
    trip_type            str | None     one_way | round_trip
    source_endpoint      str | None     month-matrix | prices_for_dates | latest | calendar | cheap
    flight_duration_min  int | None     airborne minutes when the endpoint provides them

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
    "trip_type",
    "source_endpoint",
    "flight_duration_min",
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
    "trip_type",
    "source_endpoint",
    "flight_duration_min",
]

_LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "origin": ("origin", "origin"),
    "destination": ("destination", "destination"),
    "departure_at": ("departure_at", "depart_date", "departure_at"),
    "return_at": ("return_at", "return_date", "return_at"),
    "fetched_at": ("fetched_at", "fetched_at"),
    "expires_at": ("expires_at", "expires_at"),
    "flight_number": ("flight_number", "flight_number"),
    "origin_city": ("origin_city", "origin_city"),
    "destination_city": ("destination_city", "destination_city"),
    "trip_type": ("trip_type",),
    "source_endpoint": ("source_endpoint",),
    "flight_duration_min": ("flight_duration_min", "duration"),
}


def canonicalize_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    """Map legacy / alternate JSON keys onto the current snapshot schema."""
    out = dict(row)
    for canonical, aliases in _LEGACY_ALIASES.items():
        if out.get(canonical) not in (None, ""):
            continue
        for alias in aliases:
            value = row.get(alias)
            if value not in (None, ""):
                out[canonical] = value
                break
    if not out.get("trip_type"):
        out["trip_type"] = "round_trip" if out.get("return_at") not in (None, "") else "one_way"
    return out


def normalize_cheapest_prices(
    payload: dict[str, Any],
    origin: str,
    currency: str,
    trip_type: str | None = None,
    source_endpoint: str = "cheap",
) -> list[dict[str, Any]]:
    """Flattens a `/v1/prices/cheap` response into snapshot dicts.

    Response shape: `data -> {destination: {stop_count: offer}}`, where
    `stop_count` ("0", "1", "2", ...) is the dict key.
    """
    fetched_at = _now_iso()
    snapshots: list[dict[str, Any]] = []

    for destination, offers_by_stops in (payload.get("data") or {}).items():
        if not isinstance(offers_by_stops, dict):
            continue
        for stop_count, offer in offers_by_stops.items():
            if not isinstance(offer, dict):
                continue
            snapshots.append(
                _base_snapshot(
                    origin=origin,
                    destination=destination,
                    offer=offer,
                    currency=currency,
                    fetched_at=fetched_at,
                    stops=_safe_int(stop_count),
                    trip_type=trip_type or _infer_trip_type(offer.get("return_at")),
                    source_endpoint=source_endpoint,
                )
            )
    return snapshots


def normalize_calendar_prices(
    payload: dict[str, Any],
    origin: str,
    destination: str,
    currency: str,
    trip_type: str | None = None,
    source_endpoint: str = "calendar",
) -> list[dict[str, Any]]:
    """Flattens a `/v1/prices/calendar` response into snapshot dicts.

    Response shape: `data -> {yyyy-mm-dd: offer}`, one cheapest offer per
    calendar day. Stop count lives in `transfers` or `transfers`.
    """
    fetched_at = _now_iso()
    snapshots: list[dict[str, Any]] = []

    for _day, offer in (payload.get("data") or {}).items():
        if not isinstance(offer, dict):
            continue
        snapshots.append(
            _base_snapshot(
                origin=offer.get("origin", origin),
                destination=offer.get("destination", destination),
                offer=offer,
                currency=currency,
                fetched_at=fetched_at,
                stops=_first_present(offer, "transfers", "transfers", "number_of_changes"),
                trip_type=trip_type or _infer_trip_type(offer.get("return_at")),
                source_endpoint=source_endpoint,
            )
        )
    return snapshots


def normalize_latest_prices(
    payload: dict[str, Any],
    currency: str | None = None,
    trip_type: str | None = None,
    source_endpoint: str = "latest",
) -> list[dict[str, Any]]:
    """Flattens a `/v2/prices/latest` response into snapshot dicts.

    Price maps from `value`. Airline and flight number are usually absent.
    """
    fetched_at = _now_iso()
    snapshots: list[dict[str, Any]] = []

    for offer in payload.get("data") or []:
        if not isinstance(offer, dict):
            continue
        return_at = _blank_to_none(offer.get("return_date") or offer.get("return_at"))
        snapshots.append(
            {
                "origin": _upper(offer.get("origin")),
                "destination": _upper(offer.get("destination")),
                "departure_at": offer.get("depart_date") or offer.get("departure_at"),
                "return_at": return_at,
                "airline": offer.get("airline"),
                "flight_number": _safe_int(offer.get("flight_number")),
                "price": offer.get("value") if offer.get("value") is not None else offer.get("price"),
                "currency": (currency or offer.get("currency") or "").upper() or None,
                "stops": _first_present(offer, "number_of_changes", "number_of_changes", "transfers"),
                "expires_at": offer.get("expires_at"),
                "fetched_at": fetched_at,
                "source": "travelpayouts",
                "trip_type": trip_type or _infer_trip_type(return_at),
                "source_endpoint": source_endpoint,
                "flight_duration_min": _safe_int(offer.get("duration")),
            }
        )
    return snapshots


def normalize_month_matrix(
    payload: dict[str, Any],
    origin: str,
    destination: str,
    currency: str,
    trip_type: str,
    source_endpoint: str = "month-matrix",
) -> list[dict[str, Any]]:
    """Flattens a `/v2/prices/month-matrix` response into snapshot dicts."""
    fetched_at = _now_iso()
    snapshots: list[dict[str, Any]] = []

    for offer in payload.get("data") or []:
        if not isinstance(offer, dict):
            continue
        return_at = _blank_to_none(offer.get("return_date") or offer.get("return_at"))
        snapshots.append(
            {
                "origin": _upper(offer.get("origin") or origin),
                "destination": _upper(offer.get("destination") or destination),
                "departure_at": offer.get("depart_date") or offer.get("departure_at"),
                "return_at": return_at,
                "airline": offer.get("airline"),
                "flight_number": _safe_int(offer.get("flight_number")),
                "price": offer.get("value") if offer.get("value") is not None else offer.get("price"),
                "currency": currency.upper(),
                "stops": _first_present(offer, "number_of_changes", "number_of_changes", "transfers"),
                "expires_at": offer.get("expires_at"),
                "fetched_at": fetched_at,
                "source": "travelpayouts",
                "trip_type": trip_type,
                "source_endpoint": source_endpoint,
                "flight_duration_min": _safe_int(offer.get("duration")),
            }
        )
    return snapshots


def normalize_prices_for_dates(
    payload: dict[str, Any],
    origin: str,
    destination: str,
    currency: str,
    trip_type: str,
    source_endpoint: str = "prices_for_dates",
) -> list[dict[str, Any]]:
    """Flattens an `/aviasales/v3/prices_for_dates` response into snapshot dicts."""
    fetched_at = _now_iso()
    snapshots: list[dict[str, Any]] = []
    currency_value = (payload.get("currency") or currency or "usd").upper()

    for offer in payload.get("data") or []:
        if not isinstance(offer, dict):
            continue
        return_at = _blank_to_none(offer.get("return_at") or offer.get("return_date"))
        snapshots.append(
            {
                "origin": _upper(offer.get("origin") or origin),
                "destination": _upper(offer.get("destination") or destination),
                "departure_at": offer.get("departure_at") or offer.get("depart_date"),
                "return_at": return_at,
                "airline": offer.get("airline"),
                "flight_number": _safe_int(offer.get("flight_number")),
                "price": offer.get("price") if offer.get("price") is not None else offer.get("value"),
                "currency": currency_value,
                "stops": _first_present(offer, "transfers", "transfers", "number_of_changes"),
                "expires_at": offer.get("expires_at"),
                "fetched_at": fetched_at,
                "source": "travelpayouts",
                "trip_type": trip_type,
                "source_endpoint": source_endpoint,
                "flight_duration_min": _safe_int(offer.get("duration")),
            }
        )
    return snapshots


def filter_snapshots(
    snapshots: list[dict[str, Any]],
    *,
    today: datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop rows missing required fields or with a departure already in the past.

    Expired quotes are kept (they are still the price at fetch time). The
    returned stats include how many were expired vs dropped.
    """
    now = today or datetime.now(timezone.utc)
    today_date = now.date()
    kept: list[dict[str, Any]] = []
    dropped = 0
    expired = 0

    for raw in snapshots:
        row = canonicalize_snapshot(raw)
        origin = row.get("origin")
        destination = row.get("destination")
        price = row.get("price")
        departure_at = row.get("departure_at")
        if not origin or not destination or price is None or not departure_at:
            dropped += 1
            continue
        dep = pd.to_datetime(departure_at, utc=True, errors="coerce")
        if pd.isna(dep) or dep.date() < today_date:
            dropped += 1
            continue
        expires_at = row.get("expires_at")
        if expires_at:
            exp = pd.to_datetime(expires_at, utc=True, errors="coerce")
            fetched = pd.to_datetime(row.get("fetched_at"), utc=True, errors="coerce")
            if not pd.isna(exp) and not pd.isna(fetched) and exp < fetched:
                expired += 1
        if row.get("trip_type") == "one_way":
            row["return_at"] = None
        kept.append(row)

    return kept, {"dropped": dropped, "expired": expired, "kept": len(kept)}


def snapshots_to_dataframe(
    snapshots: list[dict[str, Any]],
    add_city_names: bool = True,
) -> pd.DataFrame:
    """Converts a list of flat snapshot dicts into a typed pandas DataFrame."""
    rows = [canonicalize_snapshot(s) for s in snapshots]
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    for col in ("departure_at", "return_at", "expires_at", "fetched_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["stops"] = pd.to_numeric(df["stops"], errors="coerce").astype("Int64")
    df["flight_number"] = pd.to_numeric(df["flight_number"], errors="coerce").astype("Int64")
    df["flight_duration_min"] = pd.to_numeric(df["flight_duration_min"], errors="coerce").astype("Int64")

    for col in ("origin", "destination", "airline", "currency", "source", "trip_type", "source_endpoint"):
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

    Appends by default. If the existing file's header does not match the
    current schema, the file is rewritten (old rows + new rows) instead of
    appending misaligned columns.
    """
    path = Path(path)
    if not snapshots:
        return path

    df = snapshots_to_dataframe(snapshots)

    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        existing = _canonicalize_dataframe(existing)
        if list(existing.columns) != list(df.columns):
            logger.warning(
                "CSV header mismatch at %s; rewriting with the current schema",
                path,
            )
            old_rows = existing.to_dict(orient="records")
            df = snapshots_to_dataframe(
                [canonicalize_snapshot(r) for r in old_rows] + snapshots
            )
            append = False
        else:
            # Keep datetime columns consistent when appending.
            pass

    file_exists = path.exists() and path.stat().st_size > 0
    write_header = not (append and file_exists)
    mode = "a" if append and file_exists else "w"
    df.to_csv(path, mode=mode, header=write_header, index=False)
    return path


def _canonicalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for canonical, aliases in _LEGACY_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns and alias != canonical:
                rename[alias] = canonical
                break
    if rename:
        df = df.rename(columns=rename)
    return df


def _base_snapshot(
    *,
    origin: str,
    destination: str,
    offer: dict[str, Any],
    currency: str,
    fetched_at: str,
    stops: int | None,
    trip_type: str | None,
    source_endpoint: str,
) -> dict[str, Any]:
    return_at = _blank_to_none(offer.get("return_at") or offer.get("return_date"))
    return {
        "origin": str(origin).upper(),
        "destination": str(destination).upper(),
        "departure_at": offer.get("departure_at") or offer.get("depart_date"),
        "return_at": return_at,
        "airline": offer.get("airline"),
        "flight_number": offer.get("flight_number"),
        "price": offer.get("price") if offer.get("price") is not None else offer.get("value"),
        "currency": currency.upper(),
        "stops": _safe_int(stops),
        "expires_at": offer.get("expires_at"),
        "fetched_at": fetched_at,
        "source": "travelpayouts",
        "trip_type": trip_type,
        "source_endpoint": source_endpoint,
        "flight_duration_min": _safe_int(offer.get("duration")),
    }


def _first_present(offer: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if offer.get(key) is not None:
            return _safe_int(offer.get(key))
    return None


def _infer_trip_type(return_at: Any) -> str:
    return "round_trip" if return_at not in (None, "") else "one_way"


def _blank_to_none(value: Any) -> Any:
    if value in ("", None):
        return None
    return value


def _upper(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).upper()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
