"""Scheduled collector for flight price snapshots.

Supports both cloud cron execution (GitHub Actions / OS cron) and a local
background polling loop:

    # Single run and exit (used by GitHub Actions every 12h)
    python -m src.ingestion.scheduler --once

    # Continuous background loop (uses interval_hours from config/fetch.json)
    python -m src.ingestion.scheduler
    python -m src.ingestion.scheduler --interval-hours 12

Each cycle:
    1. Acquires a lock file (`data/.scheduler.lock`) so overlapping runs skip cleanly
    2. Calls `fetch_multi.run_fetch()` across `config/routes.json`
    3. Saves the raw snapshot batch to `data/snapshots/<timestamp>.json`
    4. Appends only new/changed offers to `data/snapshots.csv`
    5. Appends a structured run record to `data/logs/runs.jsonl`
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

from src.ingestion.api_client import TravelpayoutsClient
from src.ingestion.fetch_multi import (
    DEFAULT_CSV_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROUTES_PATH,
    load_routes,
    run_fetch,
)
from src.ingestion.fetch_snapshots import DEFAULT_FETCH_CONFIG_PATH, load_fetch_config
from src.ingestion.normalizer import write_snapshots_csv

logger = logging.getLogger(__name__)

DEFAULT_LOCK_PATH = Path("data/.scheduler.lock")
DEFAULT_RUN_LOG_PATH = Path("data/logs/runs.jsonl")
STALE_LOCK_SECONDS = 2 * 60 * 60  # 2 hours


class SchedulerLockError(RuntimeError):
    """Raised when another scheduler run is already holding the lock."""


@contextlib.contextmanager
def acquire_lock(
    lock_path: Path = DEFAULT_LOCK_PATH,
    stale_after_seconds: int = STALE_LOCK_SECONDS,
) -> Iterator[None]:
    """File lock with stale-lock recovery so overlapping runs never collide."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        age = time.time() - lock_path.stat().st_mtime
        if age < stale_after_seconds and _lock_pid_alive(lock_path):
            raise SchedulerLockError(
                f"Active lock at {lock_path} (age {age:.0f}s); skipping overlapping cycle"
            )
        logger.warning("Removing stale lock at %s (age %.0fs)", lock_path, age)
        lock_path.unlink(missing_ok=True)

    payload = {
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    lock_path.write_text(json.dumps(payload))
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _lock_pid_alive(lock_path: Path) -> bool:
    try:
        data = json.loads(lock_path.read_text())
        pid = int(data.get("pid") or 0)
    except (OSError, ValueError, TypeError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _count_csv_data_rows(csv_path: Path) -> int:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = sum(1 for _ in f)
    return max(0, lines - 1)


def run_cycle(
    client: TravelpayoutsClient,
    routes: list[dict[str, str]],
    fetch_cfg: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    csv_path: Path = DEFAULT_CSV_PATH,
    run_log_path: Path = DEFAULT_RUN_LOG_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
    save_json: bool = True,
    save_csv: bool = True,
    dedupe_csv: bool = True,
) -> dict[str, Any]:
    """Executes one complete fetch -> persist -> log cycle."""
    started_at = datetime.now(timezone.utc)
    start_ts = time.time()

    with acquire_lock(lock_path):
        result = run_fetch(
            client=client,
            routes=routes,
            fetch_cfg=fetch_cfg,
            currency=fetch_cfg.get("currency"),
        )
        snapshots = result["snapshots"]
        summary = result["summary"]
        errors = result["errors"]

        snapshot_file: str | None = None
        if save_json and snapshots:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{int(start_ts)}.json"
            out_path.write_text(json.dumps(snapshots, indent=2))
            snapshot_file = str(out_path)
            logger.info("Wrote %d raw snapshot(s) to %s", len(snapshots), out_path)

        csv_rows_added = 0
        if save_csv and snapshots:
            before_rows = _count_csv_data_rows(csv_path)
            write_snapshots_csv(snapshots, csv_path, append=True, dedupe=dedupe_csv)
            after_rows = _count_csv_data_rows(csv_path)
            csv_rows_added = max(0, after_rows - before_rows)
            logger.info(
                "CSV %s: +%d new/changed row(s) (total %d)",
                csv_path,
                csv_rows_added,
                after_rows,
            )

        finished_at = datetime.now(timezone.utc)
        duration_seconds = round(time.time() - start_ts, 2)

        run_record: dict[str, Any] = {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
            "routes": summary["routes"],
            "months": summary["months"],
            "fetched": summary["fetched"],
            "unique_in_run": summary["unique_in_run"],
            "csv_rows_added": csv_rows_added,
            "failed_routes": summary["failed_routes"],
            "trip_types": summary["trip_types"],
            "snapshot_file": snapshot_file,
            "errors": errors,
        }

        if run_log_path:
            run_log_path.parent.mkdir(parents=True, exist_ok=True)
            with run_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(run_record) + "\n")

        return run_record


def main() -> None:
    import argparse

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run a single fetch cycle and exit")
    parser.add_argument("--interval-hours", type=float, default=None, help="Override interval_hours from config/fetch.json")
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES_PATH)
    parser.add_argument("--fetch-config", type=Path, default=DEFAULT_FETCH_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--run-log", type=Path, default=DEFAULT_RUN_LOG_PATH)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--no-dedupe-csv", action="store_true", help="Append all rows to CSV even if unchanged")
    args = parser.parse_args()

    token = os.environ.get("TRAVELPAYOUTS_TOKEN")
    if not token:
        raise SystemExit(
            "Set TRAVELPAYOUTS_TOKEN in your environment, .env file, or GitHub Actions secret."
        )

    fetch_cfg = load_fetch_config(args.fetch_config)
    interval_hours = float(args.interval_hours or fetch_cfg.get("interval_hours") or 12)
    routes = load_routes(args.routes)
    market = os.environ.get("TRAVELPAYOUTS_MARKET") or fetch_cfg.get("market") or "in"
    client = TravelpayoutsClient(token=token, market=market)

    while True:
        try:
            record = run_cycle(
                client=client,
                routes=routes,
                fetch_cfg=fetch_cfg,
                output_dir=args.output_dir,
                csv_path=args.csv_path,
                run_log_path=args.run_log,
                lock_path=args.lock_path,
                dedupe_csv=not args.no_dedupe_csv,
            )
            print(json.dumps(record, indent=2))
            if args.once and record["routes"] > 0 and record["failed_routes"] >= record["routes"]:
                raise SystemExit("All configured routes failed during scheduled fetch.")
        except SchedulerLockError as exc:
            logger.warning("%s", exc)
            if args.once:
                return

        if args.once:
            return

        sleep_seconds = max(60.0, interval_hours * 3600.0)
        logger.info("Sleeping %.1f hours until next cycle...", interval_hours)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
