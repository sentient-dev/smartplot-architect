"""FastAPI entrypoint for VastuVision AI."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import RLock
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
_jobs_lock = RLock()
_executor = ThreadPoolExecutor(max_workers=4)


def _run_pipeline(job_id: UUID) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.status = JobStatus.running
        job.updated_at = datetime.now(UTC)
        request = job.request

    try:
        environmental = _environmental.fetch_environmental_profile(request.location)
        decisions = _orchestrator.execute(request, environmental)
        validation: ValidationReport = _validator.evaluate(request, environmental, decisions)
        files, summary = _processor.build_outputs(request, decisions, validation.structural_score)
        with _jobs_lock:
            current = _jobs.get(job_id)
            if not current:
                return
            current.result = DesignResult(files=files, summary=summary, design_decisions=decisions, validation=validation)
            current.status = JobStatus.completed
            current.updated_at = datetime.now(UTC)
    except ExternalServiceError as exc:
        logger.exception("Pipeline failed due to environmental data error")
        with _jobs_lock:
            current = _jobs.get(job_id)
            if not current:
                return
            current.status = JobStatus.failed
            current.error = str(exc)
            current.updated_at = datetime.now(UTC)
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.exception("Pipeline failed unexpectedly")
        with _jobs_lock:
            current = _jobs.get(job_id)
            if not current:
                return
            current.status = JobStatus.failed
            current.error = f"Unexpected failure: {exc}"
            current.updated_at = datetime.now(UTC)


def _submit_pipeline(job_id: UUID) -> None:
    try:
        _executor.submit(_run_pipeline, job_id)
    except RuntimeError as exc:
        logger.exception("Failed to enqueue pipeline execution")
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return
            job.status = JobStatus.failed
            job.error = f"Failed to enqueue job: {exc}"
            job.updated_at = datetime.now(UTC)


@app.post("/api/design/analyze-plot", response_model=dict)
def analyze_plot(request: AnalyzePlotRequest) -> dict:
    job = JobRecord(request=request)
    with _jobs_lock:
        _jobs[job.job_id] = job
    _submit_pipeline(job.job_id)
    return {"job_id": str(job.job_id), "status": JobStatus.pending}


@app.get("/api/design/{job_id}/status", response_model=dict)
def get_status(job_id: UUID) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": str(job_id), "status": job.status, "error": job.error}


@app.get("/api/design/{job_id}/result", response_model=DesignResult)
def get_result(job_id: UUID) -> DesignResult:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.completed or not job.result:
        raise HTTPException(status_code=409, detail="Job not completed")
    return job.result


@app.post("/api/design/{job_id}/regenerate", response_model=dict)
def regenerate(job_id: UUID, request: RegenerateRequest) -> dict:
    with _jobs_lock:
        existing = _jobs.get(job_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Job not found")
        existing.request = existing.request.model_copy(update={"requirements": request.requirements})
        existing.error = None
        existing.result = None
        existing.status = JobStatus.pending
        existing.updated_at = datetime.now(UTC)
    _submit_pipeline(job_id)
    return {"job_id": str(job_id), "status": JobStatus.pending}


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
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="Validation report not found")
    return job.result.validation


@app.on_event("shutdown")
def shutdown_executor() -> None:
    _executor.shutdown(wait=True)


def clear_jobs_for_testing() -> None:
    """Test utility to clear in-memory jobs safely."""
    with _jobs_lock:
        _jobs.clear()
