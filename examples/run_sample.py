#!/usr/bin/env python3
"""Run the documented Bangalore sample through the design pipeline and print the result."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import _run_pipeline, _jobs, _jobs_lock  # noqa: E402
from src.models.schemas import AnalyzePlotRequest, JobRecord  # noqa: E402


def main() -> int:
    sample_path = ROOT / "examples" / "sample_input.json"
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    request = AnalyzePlotRequest.model_validate(payload)
    job = JobRecord(job_id=uuid4(), request=request)
    with _jobs_lock:
        _jobs[job.job_id] = job
    _run_pipeline(job.job_id)
    with _jobs_lock:
        completed = _jobs[job.job_id]
    if completed.status.value != "completed" or completed.result is None:
        print(json.dumps({"status": completed.status, "error": completed.error}, indent=2, default=str))
        return 1
    result = completed.result.model_dump(mode="json")
    print(json.dumps(result, indent=2))
    artifacts_ok = True
    for relative in result["files"].values():
        disk_path = ROOT / relative.lstrip("/")
        exists = disk_path.is_file() and disk_path.stat().st_size > 0
        print(f"{relative}: {'ok' if exists else 'MISSING'} ({disk_path.stat().st_size if exists else 0} bytes)")
        artifacts_ok = artifacts_ok and exists
    return 0 if artifacts_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
