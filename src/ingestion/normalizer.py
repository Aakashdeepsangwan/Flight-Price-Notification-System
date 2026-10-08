"""Normalizes raw Travelpayouts API payloads into the internal price-snapshot schema.

Internal snapshot schema (one dict per row -> maps 1:1 to the TimescaleDB
hypertable described in `src/storage/schema.sql`):

    origin               str            IATA code of the departure city
    destination          str            IATA code of the destination city
    departure_at         str | None     ISO 8601 departure timestamp
    return_at            str | None     ISO 8601 return timestamp (None for one-way)
    departure_date       str | None     LOCAL travel day (yyyy-mm-dd) of departure, from the raw text;
                                        departure_at is converted to UTC, which moves evening flights
                                        to the next day -> use this column for anything about the day
    return_date          str | None     LOCAL day of the return flight (time zone of the return city)
    departure_hour       int | None     LOCAL departure hour (0-23); None when only the day is known
    airline              str | None     IATA airline code
    flight_number        int | None
    price                float
    currency             str | None
    stops                int | None     number of layovers (0 = non-stop)
    return_stops         int | None     layovers on the way back (round-trip, prices_for_dates only)
    max_stops            int | None     higher of way-there / way-back stops (round-trip, prices_for_dates
                                        and month-matrix only); month-matrix `stops` already means this
    expires_at           str | None     when the quoted price is no longer reliable
    fetched_at           str            ISO 8601 timestamp of when *we* polled the API
    found_at             str | None     when the price was actually seen (exact time for
                                        month-matrix / latest, day only for prices_for_dates)
    source               str            always "travelpayouts" for now
    trip_type            str | None     one_way | round_trip
    source_endpoint      str | None     month-matrix | prices_for_dates | latest | calendar | cheap
    flight_duration_min  int | None     airborne minutes when the endpoint provides them
    origin_airport       str | None     IATA airport code, e.g. YHM inside city YTO (prices_for_dates only)
    destination_airport  str | None

`snapshots_to_dataframe()` additionally attaches `origin_city` /
`destination_city` (full city names, e.g. "DEL" -> "New Delhi") by joining
against Travelpayouts' own reference data — see `reference.py`.
"""

from __future__ import annotations

import logging
import re
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
    "departure_date",
    "return_date",
    "stay_days",
    "departure_hour",
    "airline",
    "flight_number",
    "price",
    "currency",
    "stops",
    "return_stops",
    "max_stops",
    "expires_at",
    "fetched_at",
    "found_at",
    "source",
    "trip_type",
    "source_endpoint",
    "flight_duration_min",
    "origin_airport",
    "destination_airport",
]

DISPLAY_COLUMNS = [
    "origin",
    "origin_city",
    "destination",
    "destination_city",
    "departure_at",
    "return_at",
    "departure_date",
    "return_date",
    "stay_days",
    "departure_hour",
    "airline",
    "flight_number",
    "price",
    "currency",
    "stops",
    "return_stops",
    "max_stops",
    "expires_at",
    "fetched_at",
    "found_at",
    "source",
    "trip_type",
    "source_endpoint",
    "flight_duration_min",
    "origin_airport",
    "destination_airport",
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
    # Local day/hour come from the raw text, before any UTC conversion.
    if out.get("departure_date") in (None, ""):
        out["departure_date"] = _local_date(out.get("departure_at"))
    if out.get("return_date") in (None, ""):
        out["return_date"] = _local_date(out.get("return_at"))
    if out.get("departure_hour") in (None, ""):
        out["departure_hour"] = _local_hour(out.get("departure_at"))
    if out.get("source_endpoint") == "month-matrix" and not (_safe_int(out.get("flight_duration_min")) or 0) > 0:
        # duration 0 placeholder rows (raw JSON saved before the fix): stops, duration and
        # found_at (midnight UTC of the fetch day) are unknown.
        out["stops"] = out["max_stops"] = out["flight_duration_min"] = out["found_at"] = None
    if out.get("max_stops") in (None, "") and out.get("trip_type") == "round_trip":
        if out.get("source_endpoint") == "month-matrix":
            out["max_stops"] = _safe_int(out.get("stops"))
        elif out.get("source_endpoint") == "prices_for_dates":
            out["max_stops"] = _max_stops(_safe_int(out.get("stops")), _safe_int(out.get("return_stops")))
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
                "return_stops": None,
                "max_stops": None,
                "expires_at": offer.get("expires_at"),
                "fetched_at": fetched_at,
                "found_at": offer.get("found_at"),
                "source": "travelpayouts",
                "trip_type": trip_type or _infer_trip_type(return_at),
                "source_endpoint": source_endpoint,
                "flight_duration_min": _safe_int(offer.get("duration")),
                "origin_airport": None,
                "destination_airport": None,
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
        # duration 0 = no flight behind the price: number_of_changes is a 0 placeholder,
        # not a non-stop flight, and found_at is just midnight UTC of the fetch day.
        # Keep price/dates, store stops/duration/found_at as unknown.
        duration = _safe_int(offer.get("duration"))
        has_details = bool(duration and duration > 0)
        stops = (
            _first_present(offer, "number_of_changes", "number_of_changes", "transfers")
            if has_details
            else None
        )
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
                "stops": stops,
                "return_stops": None,
                # number_of_changes on a round trip is already the higher of the two legs.
                "max_stops": stops if trip_type == "round_trip" else None,
                "expires_at": offer.get("expires_at"),
                "fetched_at": fetched_at,
                "found_at": offer.get("found_at") if has_details else None,
                "source": "travelpayouts",
                "trip_type": trip_type,
                "source_endpoint": source_endpoint,
                "flight_duration_min": duration if has_details else None,
                "origin_airport": None,
                "destination_airport": None,
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
        stops = _first_present(offer, "transfers", "transfers", "number_of_changes")
        # API sends return_transfers=0 on one-way offers too; there is no return leg.
        return_stops = _safe_int(offer.get("return_transfers")) if return_at else None
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
                "stops": stops,
                "return_stops": return_stops,
                "max_stops": _max_stops(stops, return_stops) if trip_type == "round_trip" else None,
                "expires_at": offer.get("expires_at"),
                "fetched_at": fetched_at,
                "found_at": _found_date_from_link(offer.get("link")),
                "source": "travelpayouts",
                "trip_type": trip_type,
                "source_endpoint": source_endpoint,
                "flight_duration_min": _safe_int(offer.get("duration")),
                "origin_airport": _upper(offer.get("origin_airport")),
                "destination_airport": _upper(offer.get("destination_airport")),
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


def drop_redundant_offers(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop airline-less rows that copy an offer already seen with an airline.

    month-matrix returns the same fare as prices_for_dates but without airline,
    flight_number, airport or departure time, so the exact dedupe key misses it.
    Round trips compare max_stops (month-matrix only knows the higher leg), and
    both rows must come from the same search day (found_at) to count as a copy.
    The copy's exact found_at is moved onto the detailed row before it is dropped.
    Call once per fetch run: the same fare in two runs is real history.
    """
    def loose_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("origin"),
            row.get("destination"),
            row.get("trip_type"),
            str(row.get("departure_at") or "")[:10],  # day only, ignore time
            str(row.get("return_at") or "")[:10],
            row.get("max_stops") if row.get("max_stops") is not None else row.get("stops"),
            round(float(row["price"]), 2),
            str(row.get("found_at") or "")[:10],  # same search day
        )

    detailed: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in snapshots:
        if row.get("airline"):
            detailed.setdefault(loose_key(row), []).append(row)

    kept: list[dict[str, Any]] = []
    for row in snapshots:
        matches = None if row.get("airline") else detailed.get(loose_key(row))
        if not matches:
            kept.append(row)
            continue
        # The copy has the exact time; prices_for_dates only knows the day.
        exact = str(row.get("found_at") or "")
        for match in matches:
            day_only = str(match.get("found_at") or "")
            if len(exact) > 10 and len(day_only) == 10 and exact[:10] == day_only:
                match["found_at"] = exact
    return kept


def snapshots_to_dataframe(
    snapshots: list[dict[str, Any]],
    add_city_names: bool = True,
) -> pd.DataFrame:
    """Converts a list of flat snapshot dicts into a typed pandas DataFrame."""
    rows = [canonicalize_snapshot(s) for s in snapshots]
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    for col in ("departure_at", "return_at", "expires_at", "fetched_at", "found_at"):
        # Endpoints mix date-only and full-timestamp values in the same column;
        # without an explicit format pandas infers one from the first row and
        # silently coerces every other shape to NaT.
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="ISO8601")

    for col in ("departure_date", "return_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")
    df["stay_days"] = (df["return_date"] - df["departure_date"]).dt.days.astype("Int64")
    df["departure_hour"] = pd.to_numeric(df["departure_hour"], errors="coerce").astype("Int64")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["stops"] = pd.to_numeric(df["stops"], errors="coerce").astype("Int64")
    df["return_stops"] = pd.to_numeric(df["return_stops"], errors="coerce").astype("Int64")
    df["max_stops"] = pd.to_numeric(df["max_stops"], errors="coerce").astype("Int64")
    df["flight_number"] = pd.to_numeric(df["flight_number"], errors="coerce").astype("Int64")
    df["flight_duration_min"] = pd.to_numeric(df["flight_duration_min"], errors="coerce").astype("Int64")

    for col in ("origin", "destination", "airline", "currency", "source", "trip_type", "source_endpoint",
                "origin_airport", "destination_airport"):
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
        "return_stops": None,
        "max_stops": None,
        "expires_at": offer.get("expires_at"),
        "fetched_at": fetched_at,
        "found_at": None,
        "source": "travelpayouts",
        "trip_type": trip_type,
        "source_endpoint": source_endpoint,
        "flight_duration_min": _safe_int(offer.get("duration")),
        "origin_airport": None,
        "destination_airport": None,
    }


def _first_present(offer: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if offer.get(key) is not None:
            return _safe_int(offer.get(key))
    return None


def _found_date_from_link(link: Any) -> str | None:
    """prices_for_dates has no `found_at`; its `link` carries `search_date=ddmmyyyy` instead."""
    match = re.search(r"search_date=(\d{8})", str(link or ""))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%d%m%Y").date().isoformat()
    except ValueError:
        return None


def _local_date(value: Any) -> str | None:
    """yyyy-mm-dd from a raw API value (`2027-01-31T22:45:00-05:00` or `2027-01-31`).

    Raw values always start with the local day. Text already converted to UTC by
    pandas (`2027-02-01 03:45:00+00:00`, space instead of T) is refused: its day
    can be shifted, so the local day is unknown.
    """
    text = str(value or "")
    if " " in text or not re.match(r"\d{4}-\d{2}-\d{2}(T|$)", text):
        return None
    return text[:10]


def _local_hour(value: Any) -> int | None:
    text = str(value or "")
    if _local_date(text) is None or "T" not in text:
        return None
    return _safe_int(text[11:13])


def _max_stops(stops: int | None, return_stops: int | None) -> int | None:
    if stops is None or return_stops is None:
        return None
    return max(stops, return_stops)


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
