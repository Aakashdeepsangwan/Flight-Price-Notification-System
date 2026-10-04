"""Travelpayouts Data API client.

Official docs: https://api.travelpayouts.com/documentation

Auth: pass your affiliate token via the `X-Access-Token` header (used here)
or as a `token` query parameter. Get a token at
https://www.travelpayouts.com/programs/100/tools/api

Endpoints wrapped here:
    GET /v1/prices/cheap              - cheapest non-stop / 1-stop / 2-stop offers for a route
    GET /v1/prices/calendar           - one cheapest offer per calendar day for a route+month
    GET /v2/prices/latest             - prices found by real travelers in the last 48h
    GET /v2/prices/month-matrix       - cheapest fare per day, grouped by stop count
    GET /aviasales/v3/prices_for_dates - many cached offers for a route+month
"""

from __future__ import annotations

import calendar
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.travelpayouts.com"


class TravelpayoutsAPIError(Exception):
    """Raised when the API responds with success=false or a non-recoverable HTTP error."""


@dataclass
class TravelpayoutsClient:
    """Thin, retrying wrapper around the Travelpayouts Data API.

    Reuse a single instance across a scheduler run so the underlying
    `requests.Session` keeps connections alive across polling cycles.
    """

    token: str
    market: str | None = None
    timeout: float = 10.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError(
                "A Travelpayouts API token is required. "
                "Get one at https://www.travelpayouts.com/programs/100/tools/api"
            )
        self._session = requests.Session()
        self._session.headers.update(
            {"X-Access-Token": self.token, "Accept": "application/json"}
        )

    def get_cheapest_prices(
        self,
        origin: str,
        destination: str,
        depart_date: str | None = None,
        return_date: str | None = None,
        currency: str = "usd",
        page: int | None = None,
    ) -> dict[str, Any]:
        """Cheapest non-stop / 1-stop / 2-stop offers for a route.

        Maps to `GET /v1/prices/cheap`. `depart_date`/`return_date` accept
        `yyyy-mm` or `yyyy-mm-dd`; omit both to get the cheapest upcoming
        prices regardless of date. Pass `destination="-"` to fetch cheapest
        offers to *all* known destinations from `origin`.
        """
        params: dict[str, Any] = {
            "origin": origin.upper(),
            "destination": destination.upper(),
            "currency": currency,
        }
        if depart_date:
            params["depart_date"] = depart_date
        if return_date:
            params["return_date"] = return_date
        if page:
            params["page"] = page
        if self.market:
            params["market"] = self.market

        return self._get("/v1/prices/cheap", params)

    def get_calendar_prices(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: str | None = None,
        calendar_type: str = "departure_date",
        length: int | None = None,
        currency: str = "usd",
    ) -> dict[str, Any]:
        """One cheapest offer per calendar day for a route + month.

        Maps to `GET /v1/prices/calendar`. `departure_date` is `yyyy-mm`
        (or `yyyy-mm-dd`). This is the endpoint to use for building a
        day-by-day price snapshot history per route (Phase 0 collection).
        """
        params: dict[str, Any] = {
            "origin": origin.upper(),
            "destination": destination.upper(),
            # Official examples use depart_date; some docs use departure_date.
            "depart_date": departure_date,
            "departure_date": departure_date,
            "calendar_type": calendar_type,
            "currency": currency,
        }
        if return_date:
            params["return_date"] = return_date
        if length:
            params["length"] = length
        if self.market:
            params["market"] = self.market

        return self._get("/v1/prices/calendar", params)

    def get_latest_prices(
        self,
        origin: str | None = None,
        destination: str | None = None,
        period_type: str = "year",
        beginning_of_period: str | None = None,
        one_way: bool = False,
        currency: str = "usd",
        limit: int = 30,
        page: int = 1,
        sorting: str = "price",
        trip_duration: int | None = None,
        show_to_affiliates: bool = False,
    ) -> dict[str, Any]:
        """Prices found by real travelers in the last 48 hours.

        Maps to `GET /v2/prices/latest`. Pass origin and destination to
        scope the result to one route. `show_to_affiliates=False` returns
        the full cache, not only partner-found fares.
        """
        params: dict[str, Any] = {
            "currency": currency,
            "period_type": period_type,
            "one_way": str(one_way).lower(),
            "limit": limit,
            "page": page,
            "sorting": sorting,
            "show_to_affiliates": str(show_to_affiliates).lower(),
        }
        if origin:
            params["origin"] = origin.upper()
        if destination:
            params["destination"] = destination.upper()
        if beginning_of_period:
            params["beginning_of_period"] = beginning_of_period
        if trip_duration:
            params["trip_duration"] = trip_duration
        if self.market:
            params["market"] = self.market

        return self._get("/v2/prices/latest", params)

    def get_month_matrix(
        self,
        origin: str,
        destination: str,
        month: str,
        one_way: bool = True,
        trip_duration: int | None = None,
        currency: str = "usd",
        show_to_affiliates: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Cheapest fare per day of a month, grouped by number of stops.

        Maps to `GET /v2/prices/month-matrix`. `month` is `yyyy-mm` or
        `yyyy-mm-01`. `trip_duration` is stay length in DAYS (round-trip) and
        matches exactly, so omit it unless you want one specific stay length.
        """
        month_start = _month_start(month)
        year, month_num = int(month_start[:4]), int(month_start[5:7])
        params: dict[str, Any] = {
            "origin": origin.upper(),
            "destination": destination.upper(),
            "month": month_start,
            "one_way": str(one_way).lower(),
            "currency": currency,
            "show_to_affiliates": str(show_to_affiliates).lower(),
            "limit": limit or calendar.monthrange(year, month_num)[1],
        }
        if trip_duration:
            params["trip_duration"] = trip_duration
        if self.market:
            params["market"] = self.market
        return self._get("/v2/prices/month-matrix", params)

    def get_prices_for_dates(
        self,
        origin: str,
        destination: str,
        departure_at: str,
        return_at: str | None = None,
        one_way: bool = True,
        currency: str = "usd",
        unique: bool = False,
        sorting: str = "price",
        direct: bool = False,
        limit: int = 1000,
        page: int = 1,
    ) -> dict[str, Any]:
        """Many cached offers for a route and month (or exact date).

        Maps to `GET /aviasales/v3/prices_for_dates`. `departure_at` is
        `yyyy-mm` or `yyyy-mm-dd`. Omit `return_at` for one-way.
        """
        params: dict[str, Any] = {
            "origin": origin.upper(),
            "destination": destination.upper(),
            "departure_at": departure_at,
            "one_way": str(one_way).lower(),
            "currency": currency,
            "cy": currency,
            "unique": str(unique).lower(),
            "sorting": sorting,
            "direct": str(direct).lower(),
            "limit": limit,
            "page": page,
        }
        if return_at:
            params["return_at"] = return_at
        if self.market:
            params["market"] = self.market
        return self._get("/aviasales/v3/prices_for_dates", params)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        params = {**params, "token": self.token}
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s",
                    path, attempt, self.max_retries, exc,
                )
                time.sleep(self.retry_backoff_seconds * attempt)
                continue

            if response.status_code == 429:
                wait = self.retry_backoff_seconds * attempt
                logger.warning(
                    "Rate limited on %s, backing off %.1fs (attempt %d/%d)",
                    path, wait, attempt, self.max_retries,
                )
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                last_error = TravelpayoutsAPIError(
                    f"{path} returned HTTP {response.status_code}"
                )
                time.sleep(self.retry_backoff_seconds * attempt)
                continue

            response.raise_for_status()
            payload = response.json()

            if not _payload_ok(payload):
                raise TravelpayoutsAPIError(
                    f"{path} returned an error: {payload.get('error')}"
                )

            return payload

        raise TravelpayoutsAPIError(
            f"Exhausted {self.max_retries} retries calling {path}: {last_error}"
        )


def _payload_ok(payload: Any) -> bool:
    """v1 uses `success`; v2/v3 use `success`. Some payloads only include `data`."""
    if not isinstance(payload, dict):
        return False
    if payload.get("error"):
        return False
    for key in ("success", "ok"):
        if key in payload:
            return bool(payload[key])
    return "data" in payload


def _month_start(month: str) -> str:
    """Normalize `yyyy-mm` or `yyyy-mm-dd` to the first day of that month."""
    if len(month) >= 10:
        return month[:10]
    if len(month) == 7:
        return f"{month}-01"
    raise ValueError(f"month must be yyyy-mm or yyyy-mm-dd, got {month!r}")


def _build_client_from_env() -> TravelpayoutsClient:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    token = os.environ.get("TRAVELPAYOUTS_TOKEN")
    if not token:
        raise SystemExit(
            "Set TRAVELPAYOUTS_TOKEN in your environment or .env file "
            "(see .env.example)."
        )
    return TravelpayoutsClient(token=token, market=os.environ.get("TRAVELPAYOUTS_MARKET") or None)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Fetch flight prices from the Travelpayouts Data API."
    )
    parser.add_argument("origin", help="IATA code of the departure city, e.g. DEL")
    parser.add_argument("destination", help="IATA code of the destination city, e.g. BOM")
    parser.add_argument(
        "--mode",
        choices=["cheap", "calendar"],
        default="cheap",
        help="cheap = /v1/prices/cheap, calendar = /v1/prices/calendar",
    )
    parser.add_argument("--depart-date", default=None, help="yyyy-mm or yyyy-mm-dd")
    parser.add_argument("--return-date", default=None, help="yyyy-mm or yyyy-mm-dd")
    parser.add_argument("--currency", default="usd")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = _build_client_from_env()

    if args.mode == "cheap":
        result = client.get_cheapest_prices(
            origin=args.origin,
            destination=args.destination,
            depart_date=args.depart_date,
            return_date=args.return_date,
            currency=args.currency,
        )
    else:
        if not args.depart_date:
            raise SystemExit("--depart-date (yyyy-mm) is required for --mode calendar")
        result = client.get_calendar_prices(
            origin=args.origin,
            destination=args.destination,
            departure_date=args.depart_date,
            return_date=args.return_date,
            currency=args.currency,
        )

    print(json.dumps(result, indent=2))
