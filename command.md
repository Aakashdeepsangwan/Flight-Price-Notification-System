# One route (rolling 4-month horizon, one-way + round-trip)
python -m src.ingestion.fetch_snapshots DEL BOM

# One route, single month (debug)
python -m src.ingestion.fetch_snapshots DEL BOM --depart-date 2026-09

# All routes in config/routes.json using config/fetch.json
python -m src.ingestion.fetch_multi
python -m src.ingestion.fetch_multi --horizon-months 4
