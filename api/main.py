"""FastAPI entrypoint for VastuVision AI."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import FastAPI, HTTPException

from src.agents.orchestrator import OrchestratorAgent
from src.config.settings import settings
from src.models.schemas import AnalyzePlotRequest, Coordinates, DesignResult, JobRecord, JobStatus, LocationInput, RegenerateRequest, ValidationReport
from src.processors.design_generator import DesignProcessor
from src.services.environmental import EnvironmentalService, ExternalServiceError
from src.utils.logging import configure_logging
from src.validators.scientific import ScientificValidator

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.app_version)

_environmental = EnvironmentalService()
_orchestrator = OrchestratorAgent()
_validator = ScientificValidator()
_processor = DesignProcessor()
_jobs: dict[UUID, JobRecord] = {}


def _run_pipeline(job: JobRecord) -> None:
    try:
        job.status = JobStatus.running
        environmental = _environmental.fetch_environmental_profile(job.request.location)
        decisions = _orchestrator.execute(job.request, environmental)
        validation: ValidationReport = _validator.evaluate(job.request, environmental, decisions)
        files, summary = _processor.build_outputs(job.request, decisions, validation.structural_score)
        job.result = DesignResult(files=files, summary=summary, design_decisions=decisions, validation=validation)
        job.status = JobStatus.completed
        job.updated_at = datetime.now(UTC)
    except ExternalServiceError as exc:
        logger.exception("Pipeline failed due to environmental data error")
        job.status = JobStatus.failed
        job.error = str(exc)
        job.updated_at = datetime.now(UTC)
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.exception("Pipeline failed unexpectedly")
        job.status = JobStatus.failed
        job.error = f"Unexpected failure: {exc}"
        job.updated_at = datetime.now(UTC)


@app.post("/api/design/analyze-plot", response_model=dict)
def analyze_plot(request: AnalyzePlotRequest) -> dict:
    job = JobRecord(request=request)
    _jobs[job.job_id] = job
    _run_pipeline(job)
    return {"job_id": str(job.job_id), "status": job.status}


@app.get("/api/design/{job_id}/status", response_model=dict)
def get_status(job_id: UUID) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": str(job_id), "status": job.status, "error": job.error}


@app.get("/api/design/{job_id}/result", response_model=DesignResult)
def get_result(job_id: UUID) -> DesignResult:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.completed or not job.result:
        raise HTTPException(status_code=409, detail="Job not completed")
    return job.result


@app.post("/api/design/{job_id}/regenerate", response_model=dict)
def regenerate(job_id: UUID, request: RegenerateRequest) -> dict:
    existing = _jobs.get(job_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Job not found")
    existing.request.requirements = request.requirements
    existing.error = None
    _run_pipeline(existing)
    return {"job_id": str(job_id), "status": existing.status}


@app.get("/api/environmental/sun-path", response_model=dict)
def get_sun_path(lat: float, lon: float) -> dict:
    try:
        data = _environmental.fetch_environmental_profile(
            LocationInput(address="ad-hoc", coordinates=Coordinates(lat=lat, lon=lon))
        )
    except ExternalServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"solar": data["solar"], "geolocation": data["geolocation"]}


@app.get("/api/validation/report", response_model=ValidationReport)
def get_validation_report(job_id: UUID) -> ValidationReport:
    job = _jobs.get(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="Validation report not found")
    return job.result.validation
