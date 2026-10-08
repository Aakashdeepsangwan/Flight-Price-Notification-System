"""Rebuilds `data/snapshots.csv` from the raw JSON audit logs in `data/snapshots/`.

Use this after a schema change in `normalizer.py` (e.g. new columns) so
the CSV history file gets a consistent header across every row, instead
of appending mismatched columns onto an old-schema file.

Run directly:

    python -m src.ingestion.rebuild_csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.ingestion.normalizer import canonicalize_snapshot, drop_redundant_offers, write_snapshots_csv

logger = logging.getLogger(__name__)

DEFAULT_JSON_DIR = Path("data/snapshots")
DEFAULT_CSV_PATH = Path("data/snapshots.csv")


def load_all_snapshots(json_dir: Path) -> list[dict[str, Any]]:
    all_snapshots: list[dict[str, Any]] = []
    for path in sorted(json_dir.glob("*.json")):
        rows = [canonicalize_snapshot(row) for row in json.loads(path.read_text())]
        for row in rows:
            if row.get("trip_type") == "one_way":
                row["return_stops"] = None  # raw files saved before the one-way fix
        all_snapshots.extend(drop_redundant_offers(rows))  # one file = one fetch run
    return all_snapshots


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    args = parser.parse_args()

    snapshots = load_all_snapshots(args.json_dir)
    logger.info("Loaded %d snapshot(s) from %s", len(snapshots), args.json_dir)

    write_snapshots_csv(snapshots, args.csv_path, append=False)
    logger.info("Rebuilt %s with the current schema", args.csv_path)


if __name__ == "__main__":
    main()
