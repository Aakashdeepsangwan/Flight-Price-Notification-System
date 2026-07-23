"""Fetch flight price snapshots for multiple routes in one run.

Reads the route list from `config/routes.json` (falls back to
`--origin`/`--destination` for a single ad-hoc route) and fetches a
day-by-day calendar snapshot for each via `fetch_snapshots.fetch_route_snapshots`.
This is the shape `scheduler.py` will run on a 6-12h cron once storage
(Phase 0) is wired up. Each run:

    1. writes the raw snapshots to `data/snapshots/<timestamp>.json` (audit log)
    2. appends the same snapshots to `data/snapshots.csv` (ML-ready price history)

Run directly:

    python -m src.ingestion.fetch_multi --depart-date 2026-08
    python -m src.ingestion.fetch_multi --routes config/routes.json --depart-date 2026-08
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.ingestion.api_client import TravelpayoutsClient, TravelpayoutsAPIError
from src.ingestion.fetch_snapshots import fetch_route_snapshots
from src.ingestion.normalizer import write_snapshots_csv

logger = logging.getLogger(__name__)

DEFAULT_ROUTES_PATH = Path("config/routes.json")
DEFAULT_OUTPUT_DIR = Path("data/snapshots")
DEFAULT_CSV_PATH = Path("data/snapshots.csv")


def load_routes(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return json.load(f)


def fetch_all_routes(
    client: TravelpayoutsClient,
    routes: list[dict[str, str]],
    depart_date: str | None,
    return_date: str | None,
    currency: str,
    sleep_between_calls: float = 0.5,
) -> dict[str, Any]:
    """Fetches snapshots for every route, tolerating per-route failures."""
    all_snapshots: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for i, route in enumerate(routes):
        origin, destination = route["origin"], route["destination"]
        try:
            snapshots = fetch_route_snapshots(
                client,
                origin=origin,
                destination=destination,
                depart_date=depart_date,
                return_date=return_date,
                currency=currency,
            )
            logger.info("%s -> %s: %d snapshot(s)", origin, destination, len(snapshots))
            all_snapshots.extend(snapshots)
        except TravelpayoutsAPIError as exc:
            logger.error("%s -> %s failed: %s", origin, destination, exc)
            errors.append({"origin": origin, "destination": destination, "error": str(exc)})

        if i < len(routes) - 1:
            time.sleep(sleep_between_calls)

    return {"snapshots": all_snapshots, "errors": errors}


def main() -> None:
    import argparse

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES_PATH, help="Path to a JSON list of {origin, destination}")
    parser.add_argument("--origin", default=None, help="Fetch a single ad-hoc route instead of --routes")
    parser.add_argument("--destination", default=None, help="Paired with --origin")
    parser.add_argument("--depart-date", default=None, help="yyyy-mm (calendar) or yyyy-mm-dd (cheap)")
    parser.add_argument("--return-date", default=None, help="yyyy-mm or yyyy-mm-dd")
    parser.add_argument("--currency", default="usd")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH, help="CSV history file to append snapshots to")
    parser.add_argument("--no-save", action="store_true", help="Print results only, don't write to disk")
    parser.add_argument("--no-csv", action="store_true", help="Skip writing/appending to --csv-path")
    args = parser.parse_args()

    token = os.environ.get("TRAVELPAYOUTS_TOKEN")
    if not token:
        raise SystemExit(
            "Set TRAVELPAYOUTS_TOKEN in your environment or .env file (see .env.example)."
        )

    if args.origin and args.destination:
        routes = [{"origin": args.origin, "destination": args.destination}]
    else:
        routes = load_routes(args.routes)

    client = TravelpayoutsClient(token=token, market=os.environ.get("TRAVELPAYOUTS_MARKET") or None)

    result = fetch_all_routes(
        client,
        routes=routes,
        depart_date=args.depart_date,
        return_date=args.return_date,
        currency=args.currency,
    )

    total = len(result["snapshots"])
    failed = len(result["errors"])
    logger.info(
        "Done: %d snapshot(s) across %d route(s), %d route(s) failed",
        total, len(routes), failed,
    )

    if not args.no_save and total:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"{int(time.time())}.json"
        out_path.write_text(json.dumps(result["snapshots"], indent=2))
        logger.info("Wrote snapshots to %s", out_path)

    if not args.no_csv and total:
        csv_path = write_snapshots_csv(result["snapshots"], args.csv_path, append=True)
        logger.info("Appended %d row(s) to %s", total, csv_path)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
