"""Tests for GraphExecutionError exception class and pipeline error propagation."""

import unittest
from unittest.mock import patch
from uuid import UUID

import api.main as app_main
from src.agents.orchestrator import GraphExecutionError, OrchestratorAgent
from src.models.schemas import AnalyzePlotRequest, JobStatus
from src.services.environmental import EnvironmentalService


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


class GraphExecutionErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        app_main.clear_jobs_for_testing()

    def test_graph_execution_error_is_runtime_error_subclass(self) -> None:
        self.assertTrue(issubclass(GraphExecutionError, RuntimeError))

    def test_graph_execution_error_raised_on_graph_failure(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        with patch("src.agents.orchestrator.design_graph") as mock_graph:
            mock_graph.invoke.side_effect = RuntimeError("graph crashed")
            with self.assertRaises(GraphExecutionError) as ctx:
                OrchestratorAgent().execute(req, env)
        self.assertIn("Design workflow failed to execute", str(ctx.exception))

    def test_graph_execution_error_chains_original_cause(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        original = ValueError("underlying graph error")
        with patch("src.agents.orchestrator.design_graph") as mock_graph:
            mock_graph.invoke.side_effect = original
            with self.assertRaises(GraphExecutionError) as ctx:
                OrchestratorAgent().execute(req, env)
        self.assertIs(ctx.exception.__cause__, original)

    def test_pipeline_marks_job_failed_on_graph_execution_error(self) -> None:
        with patch("api.main._submit_pipeline"):
            response = app_main.analyze_plot(_sample_request())
        job_id = UUID(response["job_id"])

        with patch("api.main._orchestrator") as mock_orch:
            mock_orch.execute.side_effect = GraphExecutionError("workflow failed")
            app_main._run_pipeline(job_id)

        status = app_main.get_status(job_id)
        self.assertEqual(status["status"], JobStatus.failed)
        self.assertIn("workflow failed", status["error"])


if __name__ == "__main__":
    unittest.main()
