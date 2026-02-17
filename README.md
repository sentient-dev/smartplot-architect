# smartplot-architect

Production-oriented starter implementation for **VastuVision AI**, a multi-agent floor-plan design platform.

## Features
- FastAPI REST endpoints for design jobs and environmental/validation queries
- Multi-agent orchestration with weighted conflict resolution (science-first)
- Environmental profile service with robust error boundaries
- Scientific validation and output artifact metadata generation
- Structured logging and typed request/response models
- Seed knowledge-base files for vastu rules, materials, and building-code minimums

## Project layout
- `/api/main.py`: FastAPI application entrypoint
- `/src/agents`: Agent orchestration
- `/src/services`: Environmental data integration layer
- `/src/validators`: Scientific checks
- `/src/processors`: Output generation
- `/src/models`: Pydantic schemas
- `/data`: Knowledge-base seed data
- `/tests`: Critical component tests

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## API endpoints
- `POST /api/design/analyze-plot`
- `GET /api/design/{job_id}/status`
- `GET /api/design/{job_id}/result`
- `POST /api/design/{job_id}/regenerate`
- `GET /api/environmental/sun-path?lat=...&lon=...`
- `GET /api/validation/report?job_id=...`

## Testing
```bash
python -m unittest tests/test_critical_components.py
```
