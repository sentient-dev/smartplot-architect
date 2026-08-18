"""End-to-end coverage for the documented Bangalore sample input."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import api.main as app_main
from src.models.schemas import AnalyzePlotRequest, JobRecord
from src.processors.design_generator import DesignProcessor
from src.services.environmental import EnvironmentalService
from src.validators.scientific import ScientificValidator


SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_input.json"


class ExampleOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        app_main.clear_jobs_for_testing()
        self._original_processor = app_main._processor
        self.payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
        self.request = AnalyzePlotRequest.model_validate(self.payload)

    def tearDown(self) -> None:
        app_main._processor = self._original_processor
        app_main.clear_jobs_for_testing()

    def test_sample_pipeline_completes_with_real_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            app_main._processor = DesignProcessor(artifacts_dir=tmp)
            job = JobRecord(job_id=uuid4(), request=self.request)
            app_main._jobs[job.job_id] = job
            app_main._run_pipeline(job.job_id)

            status = app_main.get_status(job.job_id)
            self.assertEqual(status["status"], "completed", status)
            result = app_main.get_result(job.job_id)

            self.assertEqual(len(result.design_decisions), 8)
            self.assertEqual(result.summary["total_area"], "1500.0 sq feet")
            self.assertTrue(result.validation.compliant)
            self.assertEqual(result.summary["energy_efficiency"], result.validation.energy_efficiency)
            self.assertEqual(str(result.design_id), result.files["floor_plan_2d"].split("/")[2])

            for relative in result.files.values():
                path = Path(tmp) / "/".join(relative.strip("/").split("/")[1:])
                self.assertTrue(path.is_file(), relative)
                self.assertGreater(path.stat().st_size, 0, relative)

    def test_sample_validator_accepts_solar_aligned_living_spaces(self) -> None:
        env = EnvironmentalService().fetch_environmental_profile(self.request.location)
        from src.agents.orchestrator import OrchestratorAgent

        decisions = OrchestratorAgent().execute(self.request, env)
        report = ScientificValidator().evaluate(self.request, env, decisions)
        self.assertTrue(report.compliant)
        self.assertEqual(report.sunlight_score, 8.0)
        self.assertIn(report.energy_efficiency, {"A", "A+"})
        self.assertEqual(report.issues, [])


if __name__ == "__main__":
    unittest.main()
