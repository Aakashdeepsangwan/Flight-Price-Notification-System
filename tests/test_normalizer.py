"""Fixture tests for snapshot normalizers, legacy keys, and CSV header safety."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.normalizer import (
    DISPLAY_COLUMNS,
    canonicalize_snapshot,
    filter_snapshots,
    normalize_calendar_prices,
    normalize_cheapest_prices,
    normalize_latest_prices,
    normalize_month_matrix,
    normalize_prices_for_dates,
    snapshots_to_dataframe,
    write_snapshots_csv,
)


CHEAP_PAYLOAD = {
    "success": True,
    "data": {
        "BOM": {
            "0": {
                "price": 90,
                "airline": "6E",
                "flight_number": 201,
                "departure_at": "2026-09-10T08:00:00Z",
                "return_at": None,
                "expires_at": "2026-09-10T09:00:00Z",
            },
            "1": {
                "price": 70,
                "airline": "AI",
                "flight_number": 102,
                "departure_at": "2026-09-11T08:00:00Z",
                "expires_at": "2026-09-11T09:00:00Z",
            },
        }
    },
}

CALENDAR_PAYLOAD = {
    "success": True,
    "data": {
        "2026-09-12": {
            "origin": "DEL",
            "destination": "BOM",
            "price": 88,
            "airline": "6E",
            "flight_number": 211,
            "departure_at": "2026-09-12T06:00:00Z",
            "return_at": "2026-09-19T06:00:00Z",
            "transfers": 0,
            "expires_at": "2026-09-12T07:00:00Z",
        }
    },
}

MONTH_MATRIX_PAYLOAD = {
    "success": True,
    "data": [
        {
            "origin": "DEL",
            "destination": "BOM",
            "depart_date": "2026-09-15",
            "return_date": "",
            "value": 95,
            "number_of_changes": 0,
            "airline": "6E",
            "flight_number": 220,
        }
    ],
}

PRICES_FOR_DATES_PAYLOAD = {
    "success": True,
    "currency": "usd",
    "data": [
        {
            "origin": "DEL",
            "destination": "BOM",
            "airline": "UK",
            "flight_number": 955,
            "departure_at": "2026-09-20T10:30:00Z",
            "return_at": None,
            "transfers": 1,
            "duration": 140,
            "price": 110,
        }
    ],
}

LATEST_PAYLOAD = {
    "success": True,
    "data": [
        {
            "origin": "DEL",
            "destination": "BOM",
            "depart_date": "2026-09-21",
            "return_date": "2026-09-28",
            "value": 150,
            "number_of_changes": 1,
        }
    ],
}


class NormalizerTests(unittest.TestCase):
    def test_normalize_cheapest_prices(self) -> None:
        rows = normalize_cheapest_prices(CHEAP_PAYLOAD, origin="DEL", currency="usd", trip_type="one_way")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["origin"], "DEL")
        self.assertEqual(rows[0]["destination"], "BOM")
        self.assertEqual(rows[0]["stops"], 0)
        self.assertEqual(rows[1]["stops"], 1)
        self.assertEqual(rows[0]["trip_type"], "one_way")
        self.assertEqual(rows[0]["source_endpoint"], "cheap")

    def test_normalize_calendar_uses_transfers(self) -> None:
        rows = normalize_calendar_prices(
            CALENDAR_PAYLOAD, origin="DEL", destination="BOM", currency="usd", trip_type="round_trip"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stops"], 0)
        self.assertEqual(rows[0]["trip_type"], "round_trip")
        self.assertEqual(rows[0]["source_endpoint"], "calendar")

    def test_normalize_month_matrix(self) -> None:
        rows = normalize_month_matrix(
            MONTH_MATRIX_PAYLOAD, origin="DEL", destination="BOM", currency="usd", trip_type="one_way"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["price"], 95)
        self.assertIsNone(rows[0]["return_at"])
        self.assertEqual(rows[0]["source_endpoint"], "month-matrix")

    def test_normalize_prices_for_dates(self) -> None:
        rows = normalize_prices_for_dates(
            PRICES_FOR_DATES_PAYLOAD, origin="DEL", destination="BOM", currency="usd", trip_type="one_way"
        )
        self.assertEqual(rows[0]["airline"], "UK")
        self.assertEqual(rows[0]["flight_duration_min"], 140)
        self.assertEqual(rows[0]["stops"], 1)
        self.assertEqual(rows[0]["source_endpoint"], "prices_for_dates")

    def test_normalize_latest_prices(self) -> None:
        rows = normalize_latest_prices(LATEST_PAYLOAD, currency="usd", trip_type="round_trip")
        self.assertEqual(rows[0]["price"], 150)
        self.assertEqual(rows[0]["trip_type"], "round_trip")
        self.assertEqual(rows[0]["departure_at"], "2026-09-21")

    def test_canonicalize_legacy_keys(self) -> None:
        row = canonicalize_snapshot(
            {
                "origin": "DEL",
                "destination": "BOM",
                "departure_at": "2026-09-01T00:00:00Z",
                "return_at": "2026-09-08T00:00:00Z",
                "fetched_at": "2026-08-01T00:00:00Z",
                "expires_at": "2026-08-01T01:00:00Z",
                "flight_number": 99,
                "price": 80,
            }
        )
        self.assertEqual(row["departure_at"], "2026-09-01T00:00:00Z")
        self.assertEqual(row["return_at"], "2026-09-08T00:00:00Z")
        self.assertEqual(row["fetched_at"], "2026-08-01T00:00:00Z")
        self.assertEqual(row["expires_at"], "2026-08-01T01:00:00Z")
        self.assertEqual(row["flight_number"], 99)

    def test_filter_drops_invalid_and_past(self) -> None:
        now = datetime(2026, 8, 23, tzinfo=timezone.utc)
        rows = [
            {"origin": "DEL", "destination": "BOM", "price": 10, "departure_at": "2026-09-01"},
            {"origin": "DEL", "destination": "BOM", "price": None, "departure_at": "2026-09-01"},
            {"origin": "DEL", "destination": "BOM", "price": 10, "departure_at": "2026-07-01"},
        ]
        kept, stats = filter_snapshots(rows, today=now)
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats["dropped"], 2)

    def test_snapshots_to_dataframe_without_cities(self) -> None:
        rows = normalize_cheapest_prices(CHEAP_PAYLOAD, origin="DEL", currency="usd", trip_type="one_way")
        df = snapshots_to_dataframe(rows, add_city_names=False)
        self.assertEqual(list(df.columns), DISPLAY_COLUMNS)
        self.assertEqual(len(df), 2)

    def test_write_csv_rewrites_on_header_mismatch(self) -> None:
        rows = normalize_cheapest_prices(CHEAP_PAYLOAD, origin="DEL", currency="usd", trip_type="one_way")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots.csv"
            path.write_text("origin,destination,price\nDEL,BOM,1\n")
            write_snapshots_csv(rows, path, append=True)
            text = path.read_text()
            header = text.splitlines()[0]
            self.assertIn("trip_type", header)
            self.assertIn("origin_city", header)
            self.assertGreaterEqual(len(text.splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
