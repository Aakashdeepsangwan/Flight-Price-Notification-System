"""Fetch flight price snapshots for one route across a rolling month horizon.

Run directly:

    python -m src.ingestion.fetch_snapshots DEL BOM
    python -m src.ingestion.fetch_snapshots DEL BOM --depart-date 2026-09

Reads fetch criteria from `config/fetch.json` (horizon, trip types, stay
buckets) unless `--depart-date` pins a single month. `scheduler.py` will
call `fetch_route_snapshots` on each polling cycle.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from src.ingestion.api_client import TravelpayoutsClient, TravelpayoutsAPIError
from src.ingestion.normalizer import (
    filter_snapshots,
    normalize_calendar_prices,
    normalize_cheapest_prices,
    normalize_latest_prices,
    normalize_month_matrix,
    normalize_prices_for_dates,
)

logger = logging.getLogger(__name__)

DEFAULT_FETCH_CONFIG_PATH = Path("config/fetch.json")

DEFAULT_FETCH_CONFIG: dict[str, Any] = {
    "horizon_months": 4,
    "trip_types": ["one_way", "round_trip"],
    "round_trip_stay_days": [7, 14],
    "currency": "usd",
    "sleep_seconds": 0.4,
    "market": "in",
}


def load_fetch_config(path: Path = DEFAULT_FETCH_CONFIG_PATH) -> dict[str, Any]:
    cfg = dict(DEFAULT_FETCH_CONFIG)
    if path.exists():
        with path.open() as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            cfg.update({k: v for k, v in loaded.items() if v is not None})
    return cfg


def rolling_months(horizon_months: int, today: date | None = None) -> list[str]:
    """Current month plus the next (horizon_months - 1) months, as yyyy-mm."""
    start = today or datetime.now(timezone.utc).date()
    year, month = start.year, start.month
    months: list[str] = []
    for offset in range(max(1, horizon_months)):
        m = month + offset
        y = year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        months.append(f"{y:04d}-{m:02d}")
    return months


def fetch_route_snapshots(
    client: TravelpayoutsClient,
    origin: str,
    destination: str,
    months: list[str] | None = None,
    fetch_cfg: dict[str, Any] | None = None,
    depart_date: str | None = None,
    return_date: str | None = None,
    currency: str | None = None,
) -> list[dict[str, Any]]:
    """Fetches price snapshots for one route across months and trip types.

    `depart_date` (yyyy-mm) still works as a single-month debug override.
    `return_date` is accepted for the cheap/calendar fallback only.
    """
    cfg = dict(DEFAULT_FETCH_CONFIG)
    if fetch_cfg:
        cfg.update(fetch_cfg)
    currency = currency or cfg.get("currency") or "usd"
    sleep_seconds = float(cfg.get("sleep_seconds") or 0)
    trip_types = list(cfg.get("trip_types") or ["one_way", "round_trip"])
    stay_days = [int(d) for d in (cfg.get("round_trip_stay_days") or [7, 14])]

    if depart_date and len(depart_date) == 7:
        months = [depart_date]
    elif not months:
        months = rolling_months(int(cfg.get("horizon_months") or 4))

    collected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for month in months:
        for trip_type in trip_types:
            month_rows = _fetch_month(
                client=client,
                origin=origin,
                destination=destination,
                month=month,
                trip_type=trip_type,
                currency=currency,
                stay_days=stay_days,
                sleep_seconds=sleep_seconds,
                return_date=return_date,
            )
            for row in month_rows:
                key = _dedupe_key(row)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(row)
            if not month_rows:
                logger.warning(
                    "%s -> %s %s %s: no offers",
                    origin.upper(),
                    destination.upper(),
                    month,
                    trip_type,
                )

    kept, stats = filter_snapshots(collected)
    origin_u, dest_u = origin.upper(), destination.upper()
    kept = [
        row
        for row in kept
        if str(row.get("origin") or "").upper() == origin_u
        and str(row.get("destination") or "").upper() == dest_u
    ]
    if stats["dropped"] or stats["expired"]:
        logger.info(
            "%s -> %s: kept %d, dropped %d past/invalid, %d expired quotes retained",
            origin.upper(),
            destination.upper(),
            stats["kept"],
            stats["dropped"],
            stats["expired"],
        )
    return kept


def _fetch_month(
    *,
    client: TravelpayoutsClient,
    origin: str,
    destination: str,
    month: str,
    trip_type: str,
    currency: str,
    stay_days: list[int],
    sleep_seconds: float,
    return_date: str | None,
) -> list[dict[str, Any]]:
    one_way = trip_type == "one_way"
    rows: list[dict[str, Any]] = []

    if one_way:
        payload = _call(
            client.get_month_matrix,
            sleep_seconds,
            origin=origin,
            destination=destination,
            month=month,
            one_way=True,
            currency=currency,
        )
        if payload:
            rows.extend(
                normalize_month_matrix(
                    payload, origin, destination, currency, trip_type="one_way"
                )
            )
    else:
        for stay in stay_days:
            payload = _call(
                client.get_month_matrix,
                sleep_seconds,
                origin=origin,
                destination=destination,
                month=month,
                one_way=False,
                trip_duration=max(1, stay // 7),
                currency=currency,
            )
            if payload:
                rows.extend(
                    normalize_month_matrix(
                        payload, origin, destination, currency, trip_type="round_trip"
                    )
                )

    rows.extend(
        _prices_for_dates_pages(
            client=client,
            origin=origin,
            destination=destination,
            month=month,
            one_way=one_way,
            currency=currency,
            trip_type=trip_type,
            sleep_seconds=sleep_seconds,
        )
    )

    latest = _call(
        client.get_latest_prices,
        sleep_seconds,
        origin=origin,
        destination=destination,
        period_type="month",
        beginning_of_period=f"{month}-01",
        one_way=one_way,
        currency=currency,
        limit=1000,
        show_to_affiliates=False,
    )
    if latest:
        latest_rows = normalize_latest_prices(
            latest, currency=currency, trip_type=trip_type
        )
        prefix = month
        rows.extend(
            r
            for r in latest_rows
            if str(r.get("origin") or "").upper() == origin.upper()
            and str(r.get("destination") or "").upper() == destination.upper()
            and str(r.get("departure_at") or "").startswith(prefix)
        )

    if rows:
        return rows

    lengths = [None] if one_way else stay_days
    for length in lengths:
        calendar_payload = _call(
            client.get_calendar_prices,
            sleep_seconds,
            origin=origin,
            destination=destination,
            departure_date=month,
            return_date=None if one_way else return_date,
            length=length,
            currency=currency,
        )
        if calendar_payload:
            rows.extend(
                normalize_calendar_prices(
                    calendar_payload,
                    origin=origin,
                    destination=destination,
                    currency=currency,
                    trip_type=trip_type,
                )
            )

    if rows:
        return rows

    cheap = _call(
        client.get_cheapest_prices,
        sleep_seconds,
        origin=origin,
        destination=destination,
        depart_date=month,
        return_date=None if one_way else return_date,
        currency=currency,
    )
    if cheap:
        rows.extend(
            normalize_cheapest_prices(
                cheap, origin=origin, currency=currency, trip_type=trip_type
            )
        )
        rows = [
            r
            for r in rows
            if str(r.get("destination") or "").upper() == destination.upper()
        ]
    return rows


def _prices_for_dates_pages(
    *,
    client: TravelpayoutsClient,
    origin: str,
    destination: str,
    month: str,
    one_way: bool,
    currency: str,
    trip_type: str,
    sleep_seconds: float,
    limit: int = 1000,
    max_pages: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = _call(
            client.get_prices_for_dates,
            sleep_seconds,
            origin=origin,
            destination=destination,
            departure_at=month,
            one_way=one_way,
            currency=currency,
            unique=False,
            limit=limit,
            page=page,
        )
        if not payload:
            break
        batch = normalize_prices_for_dates(
            payload, origin, destination, currency, trip_type=trip_type
        )
        rows.extend(batch)
        if len(payload.get("data") or []) < limit:
            break
    return rows


def _call(fn: Callable[..., dict[str, Any]], sleep_seconds: float, **kwargs: Any) -> dict[str, Any] | None:
    try:
        return fn(**kwargs)
    except TravelpayoutsAPIError as exc:
        logger.warning("%s failed: %s", getattr(fn, "__name__", fn), exc)
        return None
    finally:
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def _dedupe_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("origin"),
        row.get("destination"),
        row.get("departure_at"),
        row.get("return_at"),
        row.get("airline"),
        row.get("flight_number"),
        row.get("price"),
        row.get("stops"),
        row.get("trip_type"),
    )


def main() -> None:
    import argparse
    import json as json_lib

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origin", help="IATA code of the departure city, e.g. DEL")
    parser.add_argument("destination", help="IATA code of the destination city, e.g. BOM")
    parser.add_argument("--depart-date", default=None, help="yyyy-mm to fetch a single month (debug)")
    parser.add_argument("--return-date", default=None, help="yyyy-mm or yyyy-mm-dd (fallback only)")
    parser.add_argument("--currency", default=None)
    parser.add_argument("--fetch-config", type=Path, default=DEFAULT_FETCH_CONFIG_PATH)
    parser.add_argument("--horizon-months", type=int, default=None)
    args = parser.parse_args()

    token = os.environ.get("TRAVELPAYOUTS_TOKEN")
    if not token:
        raise SystemExit(
            "Set TRAVELPAYOUTS_TOKEN in your environment or .env file (see .env.example)."
        )

    fetch_cfg = load_fetch_config(args.fetch_config)
    if args.horizon_months:
        fetch_cfg["horizon_months"] = args.horizon_months
    market = os.environ.get("TRAVELPAYOUTS_MARKET") or fetch_cfg.get("market") or "in"
    client = TravelpayoutsClient(token=token, market=market)

    snapshots = fetch_route_snapshots(
        client,
        origin=args.origin,
        destination=args.destination,
        fetch_cfg=fetch_cfg,
        depart_date=args.depart_date,
        return_date=args.return_date,
        currency=args.currency,
    )

    logger.info(
        "Fetched %d snapshot(s) for %s -> %s",
        len(snapshots),
        args.origin.upper(),
        args.destination.upper(),
    )
    print(json_lib.dumps(snapshots, indent=2))


if __name__ == "__main__":
    main()
