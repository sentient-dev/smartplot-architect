# smartplot-architect

Production-oriented starter implementation for **VastuVision AI**, a multi-agent floor-plan design platform.

## Features
- FastAPI REST endpoints for design jobs and environmental/validation queries
- Multi-agent orchestration with weighted conflict resolution (science-first)
- Environmental profile service with robust error boundaries
- Scientific validation and output artifact metadata generation
- Structured logging and typed request/response models
- Seed knowledge-base files for vastu rules, materials, and building-code minimums

## Recommended agentic framework
For this project, **LangGraph** is the best fit for agent orchestration.

- It models this system's multi-step workflow as an explicit graph (environmental analysis → specialist agents → validation → outputs).
- It supports durable state, retries, and human-in-the-loop checkpoints, which are useful for long-running design jobs.
- It keeps agent coordination deterministic while still allowing LangChain components for prompt/tool integrations.

## Project layout
- `/api/main.py`: FastAPI application entrypoint
- `/src/agents`: Agent orchestration
- `/src/services`: Environmental data integration layer
- `/src/validators`: Scientific checks
- `/src/processors`: Output generation
- `/src/models`: Pydantic schemas
- `/data`: Knowledge-base seed data
- `/tests`: Critical component tests

## Extending agents with `BaseAgent`
`BaseAgent` defines the shared SmartPlot agent contract:
- metadata (`name`, `weight`) used by orchestrator ranking
- `run(payload, environmental)` interface for domain logic
- helper methods `require_environment(...)` and `result(...)` for consistent outputs

Example:
```python
from src.agents.orchestrator import BaseAgent

class LightingAgent(BaseAgent):
    name = "lighting"
    weight = 0.8

    def run(self, payload, environmental):
        self.require_environment(environmental, ("solar",))
        exposure = environmental["solar"]["preferred_exposure"]
        return self.result(
            f"Openings optimized for {exposure} daylight",
            "Balances natural light and heat gain",
            8.1,
        )
```

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
