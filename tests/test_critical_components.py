import unittest
from unittest.mock import patch
from uuid import UUID

import api.main as app_main
from src.agents.graph import DesignGraphState, build_design_graph, design_graph
from src.agents.orchestrator import BaseAgent, InteriorDesignerAgent, OrchestratorAgent
from src.agents.orchestrator import BaseAgent, OrchestratorAgent, SiteEngineerAgent
from src.agents.orchestrator import BaseAgent, ConstructionBuilderAgent, OrchestratorAgent
from src.agents.orchestrator import BaseAgent, OrchestratorAgent, VastuExpertAgent
from src.agents.orchestrator import (
    ArchitectAgent,
    BaseAgent,
    GeologistAgent,
    MeteorologistAgent,
    OrchestratorAgent,
)
from src.agents.orchestrator import ArchitectAgent, BaseAgent, MeteorologistAgent, OrchestratorAgent
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

    def test_base_agent_result_uses_subclass_metadata(self) -> None:
        class DummyAgent(BaseAgent):
            name = "dummy"
            weight = 0.6

            def run(self, payload, environmental):
                return self.result("test decision", "test reasoning", 8.0)

        result = DummyAgent().run(_sample_request(), {})
        self.assertEqual(result.name, "dummy")
        self.assertEqual(result.weight, 0.6)
        self.assertEqual(result.score, 8.0)

    def test_base_agent_accepts_weight_boundaries(self) -> None:
        class LowerBoundaryAgent(BaseAgent):
            name = "lower-boundary"
            weight = 0.0

            def run(self, payload, environmental):
                return self.result("decision", "reasoning", 7.0)

        class UpperBoundaryAgent(BaseAgent):
            name = "upper-boundary"
            weight = 1.0

            def run(self, payload, environmental):
                return self.result("decision", "reasoning", 7.0)

        self.assertEqual(LowerBoundaryAgent().weight, 0.0)
        self.assertEqual(UpperBoundaryAgent().weight, 1.0)

    def test_base_agent_validates_subclass_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "weight"):
            class InvalidWeightAgent(BaseAgent):
                name = "invalid"
                weight = 1.1

                def run(self, payload, environmental):
                    return self.result("decision", "reasoning", 7.0)

        with self.assertRaisesRegex(ValueError, "weight"):
            class InvalidNegativeWeightAgent(BaseAgent):
                name = "invalid-negative"
                weight = -0.1

                def run(self, payload, environmental):
                    return self.result("decision", "reasoning", 7.0)

        with self.assertRaisesRegex(ValueError, "weight"):
            class InvalidNonNumericWeightAgent(BaseAgent):
                name = "invalid-type"
                weight = "heavy"

                def run(self, payload, environmental):
                    return self.result("decision", "reasoning", 7.0)

    def test_base_agent_validates_subclass_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "name"):
            class InvalidEmptyNameAgent(BaseAgent):
                name = ""
                weight = 0.8

                def run(self, payload, environmental):
                    return self.result("decision", "reasoning", 7.0)

        with self.assertRaisesRegex(ValueError, "name"):
            class InvalidNonStringNameAgent(BaseAgent):
                name = 123
                weight = 0.8

                def run(self, payload, environmental):
                    return self.result("decision", "reasoning", 7.0)

    def test_base_agent_environment_guard(self) -> None:
        class EnvAwareAgent(BaseAgent):
            name = "env-aware"
            weight = 0.8

            def run(self, payload, environmental):
                self.require_environment(environmental, ("solar", "wind"))
                return self.result("decision", "reasoning", 7.0)

        with self.assertRaises(KeyError):
            EnvAwareAgent().run(_sample_request(), {"solar": {}})

        result = EnvAwareAgent().run(_sample_request(), {"solar": {}, "wind": {}})
        self.assertEqual(result.name, "env-aware")

    def test_interior_designer_agent_outputs_deterministic_space_logic(self) -> None:
        result = InteriorDesignerAgent().run(_sample_request(), {})
        self.assertEqual(result.name, "interior_designer")
        self.assertEqual(result.weight, 0.75)
        self.assertEqual(result.score, 8.1)
        self.assertIn("3BR/2BA", result.decision)
        self.assertIn("comfort", result.decision.lower())
        self.assertIn("circulation", result.decision.lower())
    def test_construction_builder_agent_generates_climate_adaptive_package(self) -> None:
        req = _sample_request()
        result = ConstructionBuilderAgent().run(
            req,
            {
                "rainfall_mm": 850.0,
                "weather": {"average_temp_c": 26.0},
            },
        )
        self.assertEqual(result.name, "construction_builder")
        self.assertEqual(result.weight, 0.9)
        self.assertIn("damp-proofed masonry", result.decision)
        self.assertIn("high-albedo insulated roof", result.decision)
        self.assertIn("batch quantity takeoff", result.decision)
        self.assertIn("Climate-adaptive material scheduling", result.reasoning)

    def test_construction_builder_agent_uses_thermal_render_for_low_rainfall(self) -> None:
        req = _sample_request()
        result = ConstructionBuilderAgent().run(
            req,
            {
                "rainfall_mm": 500.0,
                "weather": {"average_temp_c": 22.0},
            },
        )
        self.assertIn("thermally rendered masonry", result.decision)
        self.assertIn("standard insulated roof", result.decision)

    def test_construction_builder_agent_requires_weather_average_temperature(self) -> None:
        with self.assertRaisesRegex(KeyError, "weather.average_temp_c"):
            ConstructionBuilderAgent().run(
                _sample_request(),
                {"rainfall_mm": 850.0, "weather": {}},
            )

    def test_construction_builder_agent_rejects_non_numeric_weather_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "weather.average_temp_c"):
            ConstructionBuilderAgent().run(
                _sample_request(),
                {"rainfall_mm": 850.0, "weather": {"average_temp_c": "hot"}},
            )
    def test_vastu_expert_returns_special_result_when_disabled(self) -> None:
        base_req = _sample_request()
        req = base_req.model_copy(
            update={"requirements": base_req.requirements.model_copy(update={"apply_vastu": False})}
        )
        result = VastuExpertAgent().run(req, {})
        self.assertEqual(result.name, "vastu_expert")
        self.assertEqual(result.weight, 0.7)
        self.assertEqual(result.score, 0.0)
        self.assertIn("skipped", result.decision.lower())

    def test_vastu_expert_returns_tradition_adjustment_when_enabled(self) -> None:
        result = VastuExpertAgent().run(_sample_request(), {})
        self.assertEqual(result.name, "vastu_expert")
        self.assertEqual(result.weight, 0.7)
        self.assertIn("south-east", result.decision.lower())
        self.assertIn("tradition-based", result.reasoning.lower())
    def test_geologist_agent_requires_elevation(self) -> None:
        with self.assertRaisesRegex(KeyError, "elevation_m"):
            GeologistAgent().run(_sample_request(), {"solar": {}, "wind": {}})

    def test_geologist_agent_elevation_logic_is_deterministic(self) -> None:
        payload = _sample_request()
        low = GeologistAgent().run(payload, {"elevation_m": 120})
        mid = GeologistAgent().run(payload, {"elevation_m": 320})
        high = GeologistAgent().run(payload, {"elevation_m": 720})

        self.assertIn("Raised plinth foundation", low.decision)
        self.assertIn("Reinforced strip footing", mid.decision)
        self.assertIn("Stepped reinforced foundation", high.decision)
        self.assertEqual(low.score, 8.0)
        self.assertEqual(mid.score, 8.0)
        self.assertEqual(high.score, 8.0)

    def test_meteorologist_agent_requires_wind_environment_key(self) -> None:
        with self.assertRaises(KeyError):
            MeteorologistAgent().run(_sample_request(), {"solar": {}})

    def test_meteorologist_agent_uses_prevailing_wind_direction(self) -> None:
        result = MeteorologistAgent().run(
            _sample_request(),
            {"wind": {"prevailing_direction": "NE"}},
        )
        self.assertEqual(result.name, "meteorologist")
        self.assertEqual(result.weight, 0.9)
        self.assertIn("NE", result.decision)

    def test_meteorologist_agent_requires_prevailing_direction(self) -> None:
        with self.assertRaisesRegex(KeyError, "wind\\.prevailing_direction"):
            MeteorologistAgent().run(_sample_request(), {"wind": {"avg_speed_mps": 4.2}})

    def test_architect_agent_uses_solar_preferred_exposure(self) -> None:
        result = ArchitectAgent().run(
            _sample_request(),
            {"solar": {"preferred_exposure": "east"}},
        )
        self.assertEqual(result.name, "architect")
        self.assertEqual(result.weight, 1.0)
        self.assertIn("east", result.decision.lower())

    def test_architect_agent_requires_solar_data(self) -> None:
        with self.assertRaises(KeyError):
            ArchitectAgent().run(_sample_request(), {})

        with self.assertRaisesRegex(KeyError, "preferred_exposure"):
            ArchitectAgent().run(_sample_request(), {"solar": {}})


    def test_scientific_validator_produces_report(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        decisions = OrchestratorAgent().execute(req, env)
        report = ScientificValidator().evaluate(req, env, decisions)
        self.assertIn(report.energy_efficiency, {"A", "B"})
        self.assertIsInstance(report.compliant, bool)

    def test_site_engineer_uses_road_facing_for_access_logic(self) -> None:
        req = _sample_request()
        req = req.model_copy(update={"plot": req.plot.model_copy(update={"road_facing": " East "})})
        result = SiteEngineerAgent().run(req, {})
        self.assertEqual(result.name, "site_engineer")
        self.assertEqual(result.weight, 0.85)
        self.assertEqual(result.score, 7.8)
        self.assertEqual(result.decision, "Main construction gate on east edge with north-side unloading pocket")

    def test_site_engineer_fallback_is_normalized(self) -> None:
        req = _sample_request()
        req = req.model_copy(update={"plot": req.plot.model_copy(update={"road_facing": " North-East "})})
        result = SiteEngineerAgent().run(req, {})
        self.assertEqual(result.decision, "Main construction gate aligned to north-east road edge")

    @patch("api.main._submit_pipeline")
    def test_analyze_plot_enqueues_pipeline_and_returns_pending(self, submit_pipeline) -> None:
        response = app_main.analyze_plot(_sample_request())
        job_id = UUID(response["job_id"])

        self.assertEqual(response["status"], "pending")
        submit_pipeline.assert_called_once_with(job_id)
        self.assertEqual(app_main.get_status(job_id)["status"], "pending")

    def test_structural_engineer_agent_safety_first_wall_logic(self) -> None:
        from src.agents.orchestrator import StructuralEngineerAgent

        req = _sample_request()
        result = StructuralEngineerAgent().run(
            req,
            {"wind": {"avg_speed_mps": 7.6}, "rainfall_mm": 900, "elevation_m": 220},
        )
        self.assertEqual(result.name, "structural_engineer")
        self.assertIn("300mm", result.decision)
        self.assertGreaterEqual(result.score, 9.0)

    def test_structural_engineer_agent_uses_elevation_threshold(self) -> None:
        from src.agents.orchestrator import StructuralEngineerAgent

        req = _sample_request()
        result = StructuralEngineerAgent().run(
            req,
            {"wind": {"avg_speed_mps": 4.0}, "rainfall_mm": 900, "elevation_m": 650},
        )
        self.assertIn("300mm", result.decision)
        self.assertGreaterEqual(result.score, 9.0)

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


class LangGraphWorkflowTests(unittest.TestCase):
    def test_design_graph_is_compiled(self) -> None:
        from langgraph.graph.state import CompiledStateGraph

        self.assertIsInstance(design_graph, CompiledStateGraph)

    def test_build_design_graph_returns_new_compiled_graph(self) -> None:
        from langgraph.graph.state import CompiledStateGraph

        graph = build_design_graph()
        self.assertIsInstance(graph, CompiledStateGraph)

    def test_graph_produces_eight_decisions(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        initial: DesignGraphState = {
            "payload": req,
            "environmental": env,
            "agent_results": [],
            "decisions": [],
        }
        final = design_graph.invoke(initial)
        self.assertEqual(len(final["decisions"]), 8)

    def test_graph_decisions_are_ranked_descending(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        initial: DesignGraphState = {
            "payload": req,
            "environmental": env,
            "agent_results": [],
            "decisions": [],
        }
        final = design_graph.invoke(initial)
        scores = [d.score for d in final["decisions"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_graph_decisions_contain_all_agent_names(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        initial: DesignGraphState = {
            "payload": req,
            "environmental": env,
            "agent_results": [],
            "decisions": [],
        }
        final = design_graph.invoke(initial)
        agent_names = {d.agent for d in final["decisions"]}
        expected = {
            "architect", "meteorologist", "geologist", "structural_engineer",
            "site_engineer", "vastu_expert", "interior_designer", "construction_builder",
        }
        self.assertEqual(agent_names, expected)

    def test_graph_site_engineer_decision_reflects_road_facing(self) -> None:
        req = _sample_request()
        req = req.model_copy(update={"plot": req.plot.model_copy(update={"road_facing": "west"})})
        env = EnvironmentalService().fetch_environmental_profile(req.location)
    def test_graph_requires_meteorologist_prevailing_direction(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        env["wind"] = {"avg_speed_mps": env["wind"]["avg_speed_mps"]}
        initial: DesignGraphState = {
            "payload": req,
            "environmental": env,
            "agent_results": [],
            "decisions": [],
        }
        final = design_graph.invoke(initial)
        site = next((d for d in final["decisions"] if d.agent == "site_engineer"), None)
        self.assertIsNotNone(site, "Expected a decision from 'site_engineer' agent")
        self.assertEqual(site.decision, "Main construction gate on west edge with south-side unloading pocket")

    def test_graph_site_engineer_fallback_is_normalized(self) -> None:
        req = _sample_request()
        req = req.model_copy(update={"plot": req.plot.model_copy(update={"road_facing": " NORTH-EAST "})})
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        with self.assertRaisesRegex(KeyError, "wind\\.prevailing_direction"):
            design_graph.invoke(initial)

    def test_graph_requires_meteorologist_wind_section(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        env.pop("wind")
        initial: DesignGraphState = {
            "payload": req,
            "environmental": env,
            "agent_results": [],
            "decisions": [],
        }
        final = design_graph.invoke(initial)
        site = next((d for d in final["decisions"] if d.agent == "site_engineer"), None)
        self.assertIsNotNone(site, "Expected a decision from 'site_engineer' agent")
        self.assertEqual(site.decision, "Main construction gate aligned to north-east road edge")
        with self.assertRaisesRegex(KeyError, "meteorologist: wind"):
            design_graph.invoke(initial)

    def test_graph_requires_architect_solar_preferred_exposure(self) -> None:
        req = _sample_request()
        with self.assertRaises(KeyError):
            design_graph.invoke(
                {
                    "payload": req,
                    "environmental": {},
                    "agent_results": [],
                    "decisions": [],
                }
            )

        with self.assertRaisesRegex(KeyError, "preferred_exposure"):
            design_graph.invoke(
                {
                    "payload": req,
                    "environmental": {"solar": {}},
                    "agent_results": [],
                    "decisions": [],
                }
            )

    def test_orchestrator_uses_graph_internally(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        decisions = OrchestratorAgent().execute(req, env)
        self.assertEqual(len(decisions), 8)
        self.assertGreaterEqual(decisions[0].score, decisions[-1].score)

    def test_graph_construction_builder_decision_is_climate_adaptive(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        env["rainfall_mm"] = 1100.0
        env["weather"]["average_temp_c"] = 26.0
        initial: DesignGraphState = {
            "payload": req,
            "environmental": env,
            "agent_results": [],
            "decisions": [],
        }
        final = design_graph.invoke(initial)
        construction = next((d for d in final["decisions"] if d.agent == "construction_builder"), None)
        self.assertIsNotNone(construction, "Expected a decision from 'construction_builder' agent")
        self.assertIn("reinforced concrete", construction.decision.lower())
        self.assertIn("high-albedo insulated roof", construction.decision.lower())
        self.assertIn("batch quantity takeoff", construction.decision.lower())

    def test_graph_raises_when_construction_builder_environment_is_incomplete(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        env["weather"] = {}
        initial: DesignGraphState = {
            "payload": req,
            "environmental": env,
            "agent_results": [],
            "decisions": [],
        }
        with self.assertRaisesRegex(KeyError, "weather.average_temp_c"):
            design_graph.invoke(initial)

    def test_graph_raises_when_construction_builder_environment_values_are_invalid(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        env["rainfall_mm"] = "heavy"
        initial: DesignGraphState = {
            "payload": req,
            "environmental": env,
            "agent_results": [],
            "decisions": [],
        }
        with self.assertRaisesRegex(ValueError, "rainfall_mm"):
            design_graph.invoke(initial)

    def test_vastu_skipped_when_disabled(self) -> None:
        req = _sample_request()
        req = req.model_copy(
            update={"requirements": req.requirements.model_copy(update={"apply_vastu": False})}
        )
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        initial: DesignGraphState = {
            "payload": req,
            "environmental": env,
            "agent_results": [],
            "decisions": [],
        }
        final = design_graph.invoke(initial)
        vastu = next((d for d in final["decisions"] if d.agent == "vastu_expert"), None)
        self.assertIsNotNone(vastu, "Expected a decision from 'vastu_expert' agent")
        self.assertIn("skipped", vastu.decision.lower())
        self.assertEqual(vastu.score, 0.0)

    def test_graph_requires_geologist_elevation_data(self) -> None:
        req = _sample_request()
        env = EnvironmentalService().fetch_environmental_profile(req.location)
        env.pop("elevation_m")
        initial: DesignGraphState = {
            "payload": req,
            "environmental": env,
            "agent_results": [],
            "decisions": [],
        }
        with self.assertRaisesRegex(KeyError, "elevation_m"):
            design_graph.invoke(initial)

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
