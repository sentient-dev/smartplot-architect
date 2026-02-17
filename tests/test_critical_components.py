import unittest
from unittest.mock import patch
from uuid import UUID

import api.main as app_main
from src.agents.orchestrator import OrchestratorAgent
from src.models.schemas import AnalyzePlotRequest, RegenerateRequest
from src.services.environmental import EnvironmentalService
from src.validators.scientific import ScientificValidator


def _sample_request() -> AnalyzePlotRequest:
    return AnalyzePlotRequest.model_validate(
        {
            "location": {"address": "Bangalore", "coordinates": {"lat": 12.9716, "lon": 77.5946}},
            "plot": {
                "dimensions": {"length": 30, "width": 50, "unit": "feet"},
                "orientation": "north",
                "road_facing": "east",
            },
            "requirements": {
                "bedrooms": 3,
                "bathrooms": 2,
                "kitchen": 1,
                "living_room": 1,
                "dining_room": 1,
                "budget": "mid-range",
                "style": "modern",
                "apply_vastu": True,
            },
        }
    )


class CriticalComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        app_main.clear_jobs_for_testing()

    def test_environment_profile_has_required_sections(self) -> None:
        service = EnvironmentalService()
        data = service.fetch_environmental_profile(_sample_request().location)
        expected_keys = ("geolocation", "elevation_m", "weather", "solar", "wind", "rainfall_mm")
        for key in expected_keys:
            with self.subTest(key=key):
                self.assertIn(key, data)


    def test_orchestrator_returns_ranked_agent_decisions(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        decisions = OrchestratorAgent().execute(req, env)
        self.assertEqual(len(decisions), 8)
        self.assertGreaterEqual(decisions[0].score, decisions[-1].score)


    def test_scientific_validator_produces_report(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        decisions = OrchestratorAgent().execute(req, env)
        report = ScientificValidator().evaluate(req, env, decisions)
        self.assertIn(report.energy_efficiency, {"A", "B"})
        self.assertIsInstance(report.compliant, bool)

    @patch("api.main._submit_pipeline")
    def test_analyze_plot_enqueues_pipeline_and_returns_pending(self, submit_pipeline) -> None:
        response = app_main.analyze_plot(_sample_request())
        job_id = UUID(response["job_id"])

        self.assertEqual(response["status"], "pending")
        submit_pipeline.assert_called_once_with(job_id)
        self.assertEqual(app_main.get_status(job_id)["status"], "pending")

    @patch("api.main._submit_pipeline")
    def test_regenerate_resets_job_to_pending_and_requeues(self, submit_pipeline) -> None:
        created = app_main.analyze_plot(_sample_request())
        job_id = UUID(created["job_id"])
        submit_pipeline.reset_mock()

        response = app_main.regenerate(
            job_id,
            RegenerateRequest(requirements=_sample_request().requirements),
        )

        self.assertEqual(response["status"], "pending")
        submit_pipeline.assert_called_once_with(job_id)
        self.assertEqual(app_main.get_status(job_id)["status"], "pending")


if __name__ == "__main__":
    unittest.main()
