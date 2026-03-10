"""Structural agent integration coverage kept separate to avoid main-branch test conflicts."""

import unittest

from src.agents.graph import DesignGraphState, design_graph
from src.models.schemas import AnalyzePlotRequest
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


class StructuralAgentIntegrationTests(unittest.TestCase):
    def test_graph_structural_engineer_decision_uses_regional_profile(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        env["rainfall_mm"] = 1700
        initial: DesignGraphState = {
            "payload": req,
            "environmental": env,
            "agent_results": [],
            "decisions": [],
        }
        final = design_graph.invoke(initial)
        structural = next((d for d in final["decisions"] if d.agent == "structural_engineer"), None)
        self.assertIsNotNone(structural, "Expected a decision from 'structural_engineer' agent")
        self.assertIn("300mm", structural.decision)


if __name__ == "__main__":
    unittest.main()
