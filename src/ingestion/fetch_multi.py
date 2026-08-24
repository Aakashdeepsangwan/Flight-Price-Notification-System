"""Fetch flight price snapshots for every route in `config/routes.json`.

Reads fetch criteria from `config/fetch.json` (horizon, trip types, stay
buckets) so a later 6-12h scheduler can call `run_fetch()` with no date
argument. Each run:

    1. writes snapshots to `data/snapshots/<timestamp>.json` (audit log)
    2. appends them to `data/snapshots.csv` (ML-ready price history)

Run directly:

    python -m src.ingestion.fetch_multi
    python -m src.ingestion.fetch_multi --depart-date 2026-09
    python -m src.ingestion.fetch_multi --routes config/routes.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.ingestion.api_client import TravelpayoutsClient, TravelpayoutsAPIError
from src.ingestion.fetch_snapshots import (
    DEFAULT_FETCH_CONFIG_PATH,
    fetch_route_snapshots,
    load_fetch_config,
    rolling_months,
)
from src.ingestion.normalizer import write_snapshots_csv

logger = logging.getLogger(__name__)

DEFAULT_ROUTES_PATH = Path("config/routes.json")
DEFAULT_OUTPUT_DIR = Path("data/snapshots")
DEFAULT_CSV_PATH = Path("data/snapshots.csv")


def load_routes(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return json.load(f)


def run_fetch(
    client: TravelpayoutsClient,
    routes: list[dict[str, str]],
    fetch_cfg: dict[str, Any] | None = None,
    months: list[str] | None = None,
    currency: str | None = None,
) -> dict[str, Any]:
    """Fetch every configured route. Safe to call from a later scheduler."""
    cfg = load_fetch_config() if fetch_cfg is None else dict(fetch_cfg)
    currency = currency or cfg.get("currency") or "usd"
    months = months or rolling_months(int(cfg.get("horizon_months") or 4))
    trip_types = list(cfg.get("trip_types") or ["one_way", "round_trip"])

    all_snapshots: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    per_route: list[dict[str, Any]] = []

    for route in routes:
        origin, destination = route["origin"], route["destination"]
        try:
            snapshots = fetch_route_snapshots(
                client,
                origin=origin,
                destination=destination,
                months=months,
                fetch_cfg=cfg,
                currency=currency,
            )
        except TravelpayoutsAPIError as exc:
            logger.error("%s -> %s failed: %s", origin, destination, exc)
            errors.append({"origin": origin, "destination": destination, "error": str(exc)})
            per_route.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "count": 0,
                    "empty_months": list(months),
                    "trip_types": {},
                }
            )
            continue

        counts = Counter((s.get("trip_type") or "unknown") for s in snapshots)
        empty_month_set = [
            month
            for month in months
            if not any(str(s.get("departure_at") or "").startswith(month) for s in snapshots)
        ]
        for month in empty_month_set:
            logger.warning("%s -> %s %s: empty month", origin, destination, month)
        for trip_type in trip_types:
            if counts.get(trip_type, 0) == 0:
                logger.warning("%s -> %s: no %s offers", origin, destination, trip_type)

        logger.info(
            "%s -> %s: %d snapshot(s) (%s)",
            origin,
            destination,
            len(snapshots),
            ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        )
        per_route.append(
            {
                "origin": origin,
                "destination": destination,
                "count": len(snapshots),
                "empty_months": empty_month_set,
                "trip_types": dict(counts),
            }
        )
        all_snapshots.extend(snapshots)

    unique_in_run = len({_row_key(s) for s in all_snapshots})
    trip_split = dict(Counter((s.get("trip_type") or "unknown") for s in all_snapshots))
    summary = {
        "routes": len(routes),
        "months": months,
        "fetched": len(all_snapshots),
        "unique_in_run": unique_in_run,
        "failed_routes": len(errors),
        "trip_types": trip_split,
        "per_route": per_route,
    }
    return {"snapshots": all_snapshots, "errors": errors, "summary": summary}


def fetch_all_routes(
    client: TravelpayoutsClient,
    routes: list[dict[str, str]],
    depart_date: str | None,
    return_date: str | None,
    currency: str,
    sleep_between_calls: float = 0.5,
) -> dict[str, Any]:
    """Backward-compatible wrapper used by older call sites."""
    cfg = load_fetch_config()
    cfg["sleep_seconds"] = sleep_between_calls
    months = [depart_date] if depart_date and len(depart_date) == 7 else None
    result = run_fetch(client, routes, fetch_cfg=cfg, months=months, currency=currency)
    if return_date:
        logger.info("return_date=%s is ignored by the multi-endpoint collector (stay buckets apply)", return_date)
    return result


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
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

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES_PATH, help="Path to a JSON list of {origin, destination}")
    parser.add_argument("--origin", default=None, help="Fetch a single ad-hoc route instead of --routes")
    parser.add_argument("--destination", default=None, help="Paired with --origin")
    parser.add_argument("--depart-date", default=None, help="Optional yyyy-mm to fetch a single month (debug)")
    parser.add_argument("--horizon-months", type=int, default=None, help="Override config/fetch.json horizon")
    parser.add_argument("--fetch-config", type=Path, default=DEFAULT_FETCH_CONFIG_PATH)
    parser.add_argument("--currency", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH, help="CSV history file to append snapshots to")
    parser.add_argument("--no-save", action="store_true", help="Print summary only, don't write JSON")
    parser.add_argument("--no-csv", action="store_true", help="Skip writing/appending to --csv-path")
    args = parser.parse_args()

    token = os.environ.get("TRAVELPAYOUTS_TOKEN")
    if not token:
        raise SystemExit(
            "Set TRAVELPAYOUTS_TOKEN in your environment or .env file (see .env.example)."
        )

    fetch_cfg = load_fetch_config(args.fetch_config)
    if args.horizon_months:
        fetch_cfg["horizon_months"] = args.horizon_months
    if args.currency:
        fetch_cfg["currency"] = args.currency

    if args.origin and args.destination:
        routes = [{"origin": args.origin, "destination": args.destination}]
    else:
        routes = load_routes(args.routes)

    market = os.environ.get("TRAVELPAYOUTS_MARKET") or fetch_cfg.get("market") or "in"
    client = TravelpayoutsClient(token=token, market=market)

    months = [args.depart_date] if args.depart_date and len(args.depart_date) == 7 else None
    result = run_fetch(
        client,
        routes=routes,
        fetch_cfg=fetch_cfg,
        months=months,
        currency=fetch_cfg.get("currency"),
    )

    summary = result["summary"]
    logger.info(
        "Done: %d snapshot(s) (%d unique in run) across %d route(s), %d failed; trip types %s; months %s",
        summary["fetched"],
        summary["unique_in_run"],
        summary["routes"],
        summary["failed_routes"],
        summary["trip_types"],
        summary["months"],
    )

    if not args.no_save and result["snapshots"]:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"{int(time.time())}.json"
        out_path.write_text(json.dumps(result["snapshots"], indent=2))
        logger.info("Wrote snapshots to %s", out_path)

    if not args.no_csv and result["snapshots"]:
        csv_path = write_snapshots_csv(result["snapshots"], args.csv_path, append=True)
        logger.info("Wrote %d row(s) to %s", len(result["snapshots"]), csv_path)

    print(json.dumps({"summary": summary, "errors": result["errors"]}, indent=2))


if __name__ == "__main__":
    main()
