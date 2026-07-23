"""End-to-end example: fetch flight price snapshots for a route and print them.

Run directly:

    python -m src.ingestion.fetch_snapshots DEL BOM --depart-date 2026-09

This wires `api_client.TravelpayoutsClient` (raw API calls) together with
`normalizer` (flattening into the internal snapshot schema) — the same
pair of calls `scheduler.py` will make on each polling cycle once storage
(Phase 0) is wired up.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

from src.ingestion.api_client import TravelpayoutsClient
from src.ingestion.normalizer import normalize_calendar_prices, normalize_cheapest_prices

logger = logging.getLogger(__name__)


def fetch_route_snapshots(
    client: TravelpayoutsClient,
    origin: str,
    destination: str,
    depart_date: str | None = None,
    return_date: str | None = None,
    currency: str = "usd",
) -> list[dict[str, Any]]:
    """Fetches price snapshots for one route.

    If `depart_date` is a month (`yyyy-mm`), pulls the full day-by-day
    calendar for that month via `/v1/prices/calendar` — the richer signal
    for building per-route price history. Otherwise falls back to
    `/v1/prices/cheap` for a same-day snapshot across stop counts.
    """
    if depart_date and len(depart_date) == 7:  # "yyyy-mm"
        payload = client.get_calendar_prices(
            origin=origin,
            destination=destination,
            departure_date=depart_date,
            return_date=return_date,
            currency=currency,
        )
        return normalize_calendar_prices(payload, origin=origin, destination=destination, currency=currency)

    payload = client.get_cheapest_prices(
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        return_date=return_date,
        currency=currency,
    )
    return normalize_cheapest_prices(payload, origin=origin, currency=currency)


def main() -> None:
    import argparse
    import json

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origin", help="IATA code of the departure city, e.g. DEL")
    parser.add_argument("destination", help="IATA code of the destination city, e.g. BOM")
    parser.add_argument("--depart-date", default=None, help="yyyy-mm (calendar) or yyyy-mm-dd (cheap)")
    parser.add_argument("--return-date", default=None, help="yyyy-mm or yyyy-mm-dd")
    parser.add_argument("--currency", default="usd")
    args = parser.parse_args()

    token = os.environ.get("TRAVELPAYOUTS_TOKEN")
    if not token:
        raise SystemExit(
            "Set TRAVELPAYOUTS_TOKEN in your environment or .env file (see .env.example)."
        )

    client = TravelpayoutsClient(token=token, market=os.environ.get("TRAVELPAYOUTS_MARKET") or None)

    snapshots = fetch_route_snapshots(
        client,
        origin=args.origin,
        destination=args.destination,
        depart_date=args.depart_date,
        return_date=args.return_date,
        currency=args.currency,
    )

    logger.info("Fetched %d snapshot(s) for %s -> %s", len(snapshots), args.origin.upper(), args.destination.upper())
    print(json.dumps(snapshots, indent=2))


if __name__ == "__main__":
    main()
