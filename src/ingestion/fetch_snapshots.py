"""Fetch flight price snapshots for one route across a rolling month horizon.

Run directly:

    python -m src.ingestion.fetch_snapshots DEL BOM
    python -m src.ingestion.fetch_snapshots DEL BOM --depart-date 2026-09

Reads fetch criteria from `config/fetch.json` (horizon, trip types) unless
`--depart-date` pins a single month. Every snapshot comes from
`/aviasales/v3/prices_for_dates`, the only endpoint that returns full flight
details. `scheduler.py` will call `fetch_route_snapshots` on each polling cycle.
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
from src.ingestion.normalizer import filter_snapshots, normalize_prices_for_dates

logger = logging.getLogger(__name__)

DEFAULT_FETCH_CONFIG_PATH = Path("config/fetch.json")

DEFAULT_FETCH_CONFIG: dict[str, Any] = {
    "horizon_months": 4,
    "trip_types": ["one_way", "round_trip"],
    "currency": "usd",
    "sleep_seconds": 0.4,
    "market": "in",
}


def load_fetch_config(path: Path = DEFAULT_FETCH_CONFIG_PATH) -> dict[str, Any]:
    cfg = dict(DEFAULT_FETCH_CONFIG) #make a copy of the dict to prevent the case of touching the real default 
    if path.exists():
        with path.open() as f:
            loaded = json.load(f) #convert json form to python
        if isinstance(loaded, dict):
            cfg.update({k: v for k, v in loaded.items() if v is not None}) # overwrites keys that have non-null value, if null values then it leaves the default untouched
            
    return cfg


def rolling_months(horizon_months: int, today: date | None = None) -> list[str]:
    """Current month plus the next (horizon_months - 1) months, as yyyy-mm."""
    start = today or datetime.now(timezone.utc).date() # date form (2026, 8, 26)
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
    currency: str | None = None,
) -> list[dict[str, Any]]:
    """Fetches price snapshots for one route across months and trip types.

    `depart_date` (yyyy-mm) still works as a single-month debug override.
    Round trips are fetched without a return date, so every stay length in
    the cache comes back; `stay_days` is derived from the dates downstream.
    """
    cfg = dict(DEFAULT_FETCH_CONFIG)
    if fetch_cfg:
        cfg.update(fetch_cfg)
    currency = currency or cfg.get("currency") or "usd"
    sleep_seconds = float(cfg.get("sleep_seconds") or 0)
    trip_types = list(cfg.get("trip_types") or ["one_way", "round_trip"])

    if depart_date and len(depart_date) == 7:
        months = [depart_date]
    elif not months:
        months = rolling_months(int(cfg.get("horizon_months") or 4))

    collected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for month in months:
        for trip_type in trip_types:
            month_rows = _prices_for_dates_pages(
                client=client,
                origin=origin,
                destination=destination,
                month=month,
                one_way=trip_type == "one_way",
                currency=currency,
                trip_type=trip_type,
                sleep_seconds=sleep_seconds,
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
        row.get("return_stops"),
        row.get("origin_airport"),
        row.get("destination_airport"),
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
