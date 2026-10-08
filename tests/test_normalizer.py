"""Fixture tests for snapshot normalizers, legacy keys, and CSV header safety."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ingestion.normalizer import (
    DISPLAY_COLUMNS,
    canonicalize_snapshot,
    drop_redundant_offers,
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
            "duration": 125,
            "found_at": "2026-09-01T16:13:06Z",
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
            "return_transfers": 2,
            "duration": 140,
            "price": 110,
            "origin_airport": "del",
            "destination_airport": "BOM",
            "link": "/search/DEL2009BOM1?t=UK&search_date=29082026&expected_price=110",
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
        self.assertEqual(rows[0]["found_at"], "2026-09-01T16:13:06Z")
        self.assertIsNone(rows[0]["return_stops"])
        self.assertIsNone(rows[0]["origin_airport"])

    def test_normalize_prices_for_dates(self) -> None:
        rows = normalize_prices_for_dates(
            PRICES_FOR_DATES_PAYLOAD, origin="DEL", destination="BOM", currency="usd", trip_type="one_way"
        )
        self.assertEqual(rows[0]["airline"], "UK")
        self.assertEqual(rows[0]["flight_duration_min"], 140)
        self.assertEqual(rows[0]["stops"], 1)
        self.assertEqual(rows[0]["source_endpoint"], "prices_for_dates")
        self.assertIsNone(rows[0]["return_stops"])  # one-way: return_transfers ignored
        self.assertEqual(rows[0]["found_at"], "2026-08-29")
        self.assertEqual(rows[0]["origin_airport"], "DEL")
        self.assertEqual(rows[0]["destination_airport"], "BOM")

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


class DropRedundantOffersTest(unittest.TestCase):
    def test_drops_copy_and_keeps_exact_found_at(self) -> None:
        base = {"origin": "YTO", "destination": "CUN", "trip_type": "one_way",
                "return_at": None, "price": 171, "stops": 1}
        rows = [
            {**base, "departure_at": "2026-10-25T12:15:00-04:00", "airline": "PD",
             "source_endpoint": "prices_for_dates", "found_at": "2026-10-02"},
            {**base, "departure_at": "2026-10-25", "airline": None,
             "source_endpoint": "month-matrix", "found_at": "2026-10-02T05:46:08Z"},
            {**base, "departure_at": "2026-10-25T18:00:00-04:00", "airline": "WS",
             "source_endpoint": "prices_for_dates", "found_at": "2026-10-02"},
            {**base, "departure_at": "2026-10-26", "price": 189, "stops": 0, "airline": None,
             "source_endpoint": "month-matrix", "found_at": "2026-10-03T01:00:00Z"},
        ]
        kept = drop_redundant_offers(rows)
        self.assertEqual([r["source_endpoint"] for r in kept],
                         ["prices_for_dates", "prices_for_dates", "month-matrix"])
        self.assertEqual(kept[0]["found_at"], "2026-10-02T05:46:08Z")  # exact time copied
        self.assertEqual(kept[2]["departure_at"], "2026-10-26")  # day only mm had

    def test_round_trip_matches_on_max_stops_and_found_at_day(self) -> None:
        base = {"origin": "YTO", "destination": "CUN", "trip_type": "round_trip",
                "departure_at": "2026-12-19", "return_at": "2026-12-26", "price": 753}
        pfd = {**base, "departure_at": "2026-12-19T08:00:00-05:00", "airline": "PD",
               "source_endpoint": "prices_for_dates", "stops": 0, "return_stops": 1,
               "max_stops": 1, "found_at": "2026-10-01"}
        same_search = {**base, "airline": None, "source_endpoint": "month-matrix",
                       "stops": 1, "max_stops": 1, "found_at": "2026-10-01T16:18:10Z"}
        other_search = {**same_search, "found_at": "2026-10-03T09:00:00Z"}
        kept = drop_redundant_offers([pfd, same_search, other_search])
        # 0 there + 1 back = max 1 -> same fare; a different search day is kept.
        self.assertEqual(kept, [pfd, other_search])
        self.assertEqual(pfd["found_at"], "2026-10-01T16:18:10Z")

    def test_month_matrix_duration_zero_means_unknown(self) -> None:
        payload = {"success": True, "data": [
            {"origin": "YTO", "destination": "SHA", "depart_date": "2026-11-03", "return_date": "2026-11-17",
             "value": 980, "number_of_changes": 0, "duration": 0, "found_at": "2026-10-06T00:00:00Z"},
            {"origin": "YTO", "destination": "NYC", "depart_date": "2026-11-03", "return_date": "2026-11-05",
             "value": 210, "number_of_changes": 0, "duration": 180, "found_at": "2026-10-06T08:29:29Z"},
        ]}
        unknown, nonstop = normalize_month_matrix(payload, "YTO", "SHA", "usd", trip_type="round_trip")
        self.assertIsNone(unknown["stops"])
        self.assertIsNone(unknown["max_stops"])
        self.assertIsNone(unknown["flight_duration_min"])
        self.assertEqual(unknown["price"], 980)  # price and dates are real and kept
        self.assertIsNone(unknown["found_at"])  # midnight of the fetch day, not a real search time
        self.assertEqual(nonstop["found_at"], "2026-10-06T08:29:29Z")  # real row keeps its exact time
        self.assertEqual(nonstop["stops"], 0)  # real duration -> 0 really means non-stop
        self.assertEqual(nonstop["max_stops"], 0)
        # Raw JSON saved before the fix gets cleaned the same way.
        legacy = canonicalize_snapshot({"source_endpoint": "month-matrix", "trip_type": "round_trip",
                                        "stops": 0, "max_stops": 0, "flight_duration_min": 0,
                                        "found_at": "2026-10-06T00:00:00Z"})
        self.assertIsNone(legacy["stops"])
        self.assertIsNone(legacy["max_stops"])
        self.assertIsNone(legacy["found_at"])

    def test_local_dates_survive_utc_conversion(self) -> None:
        rows = [
            # Toronto 22:45 = 03:45 UTC next day
            {"origin": "YTO", "destination": "LON", "trip_type": "one_way", "price": 600,
             "departure_at": "2027-01-31T22:45:00-05:00", "return_at": None,
             "source_endpoint": "prices_for_dates"},
            # Tokyo 08:40 return = 23:40 UTC the day before
            {"origin": "YTO", "destination": "TYO", "trip_type": "round_trip", "price": 1146,
             "departure_at": "2026-11-13T13:00:00-05:00", "return_at": "2026-11-20T08:40:00+09:00",
             "source_endpoint": "prices_for_dates"},
            {"origin": "YTO", "destination": "CUN", "trip_type": "one_way", "price": 171,
             "departure_at": "2026-10-25", "return_at": None, "source_endpoint": "month-matrix"},
        ]
        df = snapshots_to_dataframe(rows, add_city_names=False)
        self.assertEqual(df["departure_date"].dt.strftime("%Y-%m-%d").tolist(),
                         ["2027-01-31", "2026-11-13", "2026-10-25"])
        self.assertEqual(df["departure_at"].iloc[0].day, 1)  # UTC moment kept (Feb 1, 03:45)
        self.assertEqual(df["return_date"].iloc[1].strftime("%Y-%m-%d"), "2026-11-20")
        self.assertEqual(df["stay_days"].iloc[1], 7)
        self.assertTrue(pd.isna(df["return_date"].iloc[0]))
        self.assertTrue(pd.isna(df["stay_days"].iloc[0]))
        self.assertEqual(df["departure_hour"].tolist()[:2], [22, 13])
        self.assertTrue(pd.isna(df["departure_hour"].iloc[2]))  # month-matrix: day only

    def test_local_date_refuses_utc_text(self) -> None:
        # Old CSV rows hold pandas' UTC text; its day may be shifted, so leave it unknown.
        row = canonicalize_snapshot({"departure_at": "2027-02-01 03:45:00+00:00", "return_at": None})
        self.assertIsNone(row["departure_date"])
        self.assertIsNone(row["departure_hour"])

    def test_max_stops_column(self) -> None:
        rt_pfd = {"success": True, "data": [{
            "origin": "YTO", "destination": "CUN", "airline": "UA", "flight_number": 1,
            "departure_at": "2026-10-17T06:00:00-04:00", "return_at": "2026-10-24T12:00:00-04:00",
            "transfers": 2, "return_transfers": 1, "price": 408}]}
        rt_mm = {"success": True, "data": [{
            "origin": "YTO", "destination": "CUN", "depart_date": "2026-10-17",
            "return_date": "2026-10-24", "value": 408, "number_of_changes": 2, "duration": 974}]}
        pfd_row = normalize_prices_for_dates(rt_pfd, "YTO", "CUN", "usd", trip_type="round_trip")[0]
        mm_row = normalize_month_matrix(rt_mm, "YTO", "CUN", "usd", trip_type="round_trip")[0]
        self.assertEqual(pfd_row["max_stops"], 2)
        self.assertEqual(mm_row["max_stops"], 2)
        one_way = normalize_prices_for_dates(PRICES_FOR_DATES_PAYLOAD, "DEL", "BOM", "usd", trip_type="one_way")[0]
        self.assertIsNone(one_way["max_stops"])
        # Raw JSON saved before the column existed gets it filled in.
        legacy = canonicalize_snapshot({"trip_type": "round_trip", "source_endpoint": "prices_for_dates",
                                        "stops": 0, "return_stops": 1})
        self.assertEqual(legacy["max_stops"], 1)


if __name__ == "__main__":
    unittest.main()
