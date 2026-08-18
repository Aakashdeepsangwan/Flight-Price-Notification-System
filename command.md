# One Route 
python -m src.ingestion.fetch_snapshots DEL BOM --depart-date 2026-09

# All routes in Routes.json (Has all the airports in the world)
python -m src.ingestion.fetch_multi --depart-date 2026-08