"""Local cache of Travelpayouts' static city/airport reference data.

Endpoints (public, no token required):
    GET https://api.travelpayouts.com/data/en/cities.json    - IATA city code -> full city name
    GET https://api.travelpayouts.com/data/en/airports.json  - IATA airport code -> full airport name

These are large, rarely-changing static files, so we fetch once and cache
the `{code: name}` lookup on disk instead of re-downloading on every run.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

CITIES_URL = "https://api.travelpayouts.com/data/en/cities.json"
AIRPORTS_URL = "https://api.travelpayouts.com/data/en/airports.json"

DEFAULT_CACHE_DIR = Path("data/reference")
DEFAULT_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days - this reference data rarely changes


def get_city_names(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> dict[str, str]:
    """Returns a `{IATA city code: full city name}` lookup, cached on disk."""
    return _get_lookup(
        url=CITIES_URL,
        cache_path=Path(cache_dir) / "cities.json",
        ttl_seconds=ttl_seconds,
        force_refresh=force_refresh,
    )


def get_airport_names(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> dict[str, str]:
    """Returns a `{IATA airport code: full airport name}` lookup, cached on disk."""
    return _get_lookup(
        url=AIRPORTS_URL,
        cache_path=Path(cache_dir) / "airports.json",
        ttl_seconds=ttl_seconds,
        force_refresh=force_refresh,
    )


def _get_lookup(url: str, cache_path: Path, ttl_seconds: int, force_refresh: bool) -> dict[str, str]:
    if not force_refresh and cache_path.exists():
        age_seconds = time.time() - cache_path.stat().st_mtime
        if age_seconds < ttl_seconds:
            return _load_lookup(cache_path)
        logger.info("Reference cache %s is stale (%.0fh old), refreshing", cache_path, age_seconds / 3600)

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        entries: list[dict[str, Any]] = response.json()
    except requests.RequestException as exc:
        if cache_path.exists():
            logger.warning("Failed to refresh %s (%s); falling back to stale cache", url, exc)
            return _load_lookup(cache_path)
        raise

    lookup = {
        entry["code"]: entry["name"]
        for entry in entries
        if entry.get("code") and entry.get("name")
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(lookup, indent=2, ensure_ascii=False))
    logger.info("Cached %d entries to %s", len(lookup), cache_path)
    return lookup


def _load_lookup(cache_path: Path) -> dict[str, str]:
    return json.loads(cache_path.read_text())
