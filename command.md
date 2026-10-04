# One route (rolling 4-month horizon, one-way + round-trip)
python -m src.ingestion.fetch_snapshots DEL BOM

# One route, single month (debug)
python -m src.ingestion.fetch_snapshots DEL BOM --depart-date 2026-09

# All routes in config/routes.json using config/fetch.json
python -m src.ingestion.fetch_multi
python -m src.ingestion.fetch_multi --horizon-months 4




# On what basis the fetching is done : 
- Uses the Cheapest only endpoint as fallbacks

Major Factors :
- /v2/prices/month-matrix
- /aviasales/v3/prices_for_dates
- /v2/prices/latest