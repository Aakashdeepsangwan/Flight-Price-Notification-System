"""The collector must build every snapshot from /aviasales/v3/prices_for_dates only."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.ingestion.fetch_snapshots import fetch_route_snapshots
from src.ingestion.scheduler import SchedulerLockError, acquire_lock, run_cycle

ONE_WAY_OFFER = {
    "origin": "YTO", "destination": "LON", "origin_airport": "YYZ", "destination_airport": "LHR",
    "airline": "TP", "flight_number": "262", "departure_at": "2099-01-08T22:45:00-05:00",
    "transfers": 1, "return_transfers": 0, "duration": 700, "price": 277,
    "link": "/search/YTO0801LON1?search_date=05102026",
}
ROUND_TRIP_OFFER = {
    **ONE_WAY_OFFER, "return_at": "2099-01-15T10:00:00+00:00", "return_transfers": 1, "price": 540,
}


class FakeClient:
    """Answers prices_for_dates; any other endpoint call fails the test."""

    def __init__(self, months_with_data: set[str]) -> None:
        self.months_with_data = months_with_data
        self.calls: list[dict[str, Any]] = []

    def get_prices_for_dates(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        data: list[dict[str, Any]] = []
        if kwargs["departure_at"] in self.months_with_data:
            data = [ONE_WAY_OFFER if kwargs["one_way"] else ROUND_TRIP_OFFER]
        return {"success": True, "currency": "usd", "data": data}

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"collector called {name}; only prices_for_dates is allowed")


class FetchRouteSnapshotsTest(unittest.TestCase):
    def test_uses_only_prices_for_dates(self) -> None:
        client = FakeClient(months_with_data={"2099-01"})
        rows = fetch_route_snapshots(
            client, "YTO", "LON", months=["2099-01", "2099-02"], fetch_cfg={"sleep_seconds": 0}
        )
        self.assertEqual(len(client.calls), 4)  # 2 months x (one-way, round trip)
        self.assertEqual({r["source_endpoint"] for r in rows}, {"prices_for_dates"})
        self.assertEqual(sorted(r["trip_type"] for r in rows), ["one_way", "round_trip"])
        round_trip = next(r for r in rows if r["trip_type"] == "round_trip")
        self.assertEqual(round_trip["max_stops"], 1)
        self.assertNotIn("return_at", next(c for c in client.calls if not c["one_way"]))

    def test_empty_month_has_no_fallback(self) -> None:
        client = FakeClient(months_with_data=set())
        rows = fetch_route_snapshots(
            client, "YTO", "SIN", months=["2099-01"], fetch_cfg={"sleep_seconds": 0}
        )
        self.assertEqual(rows, [])
        self.assertEqual(len(client.calls), 2)


class SchedulerCycleTest(unittest.TestCase):
    def test_run_cycle_writes_json_deduped_csv_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_dir = root / "snapshots"
            csv_path = root / "snapshots.csv"
            log_path = root / "logs" / "runs.jsonl"
            lock_path = root / ".scheduler.lock"

            client = FakeClient(months_with_data={"2099-01"})
            routes = [{"origin": "YTO", "destination": "LON"}]
            cfg = {"horizon_months": 1, "trip_types": ["one_way", "round_trip"], "sleep_seconds": 0}

            # Patch rolling_months indirectly by fixing horizon_months=1; FakeClient responds to 2099-01
            from unittest.mock import patch
            with patch("src.ingestion.fetch_multi.rolling_months", return_value=["2099-01"]):
                rec1 = run_cycle(
                    client,
                    routes=routes,
                    fetch_cfg=cfg,
                    output_dir=snap_dir,
                    csv_path=csv_path,
                    run_log_path=log_path,
                    lock_path=lock_path,
                )
                self.assertEqual(rec1["fetched"], 2)
                self.assertEqual(rec1["csv_rows_added"], 2)
                self.assertFalse(lock_path.exists())

                # Second cycle with identical cached data adds 0 rows to CSV
                rec2 = run_cycle(
                    client,
                    routes=routes,
                    fetch_cfg=cfg,
                    output_dir=snap_dir,
                    csv_path=csv_path,
                    run_log_path=log_path,
                    lock_path=lock_path,
                )
                self.assertEqual(rec2["fetched"], 2)
                self.assertEqual(rec2["csv_rows_added"], 0)

            log_lines = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(len(log_lines), 2)

    def test_active_lock_blocks_overlapping_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / ".scheduler.lock"
            with acquire_lock(lock_path):
                with self.assertRaises(SchedulerLockError):
                    with acquire_lock(lock_path):
                        pass


if __name__ == "__main__":
    unittest.main()
