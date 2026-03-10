"""Multi-agent orchestration layer powered by LangGraph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from src.agents.graph import design_graph, geologist_foundation_guidance
from src.models.schemas import AnalyzePlotRequest, DesignDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResult:
    name: str
    decision: str
    reasoning: str
    score: float
    weight: float


class BaseAgent(ABC):
    """Foundation contract for all SmartPlot specialist agents.

    Subclasses are expected to:
    1. Override ``name`` and ``weight`` metadata.
    2. Implement ``run`` with their domain-specific decision logic.
    3. Use ``result`` to return consistently shaped outputs.
    """

    name = "base"
    weight = 0.5

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(cls.name, str) or not cls.name:
            raise ValueError("Agent subclasses must define a non-empty string 'name'")
        if not isinstance(cls.weight, (int, float)):
            raise ValueError("Agent subclasses must define 'weight' between 0.0 and 1.0")
        if not 0.0 <= cls.weight <= 1.0:
            raise ValueError("Agent subclasses must define 'weight' between 0.0 and 1.0")

    def require_environment(self, environmental: dict, required_keys: tuple[str, ...]) -> None:
        missing = [key for key in required_keys if key not in environmental]
        if missing:
            raise KeyError(f"Missing environmental keys for {self.name}: {', '.join(missing)}")

    def result(self, decision: str, reasoning: str, score: float) -> AgentResult:
        return AgentResult(self.name, decision, reasoning, score, self.weight)

    @abstractmethod
    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        raise NotImplementedError


class ArchitectAgent(BaseAgent):
    name = "architect"
    weight = 1.0

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        self.require_environment(environmental, ("solar",))
        preferred = environmental["solar"]["preferred_exposure"]
        return self.result(
            f"Primary living spaces aligned to {preferred}",
            "Optimized for natural daylight",
            8.5,
        )


class MeteorologistAgent(BaseAgent):
    name = "meteorologist"
    weight = 0.9

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        self.require_environment(environmental, ("wind",))
        wind = environmental["wind"]
        if "prevailing_direction" not in wind:
            raise KeyError("Missing environmental keys for meteorologist: wind.prevailing_direction")
        direction = wind["prevailing_direction"]
        return self.result(
            f"Cross-ventilation windows oriented towards {direction}",
            "Uses prevailing wind data",
            8.2,
        )


class GeologistAgent(BaseAgent):
    name = "geologist"
    weight = 0.95

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        self.require_environment(environmental, ("elevation_m",))
        elevation = environmental["elevation_m"]
        decision, reasoning = geologist_foundation_guidance(elevation)
        return self.result(
            decision,
            reasoning,
            8.0,
        )


class StructuralEngineerAgent(BaseAgent):
    name = "structural_engineer"
    weight = 1.0

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        self.require_environment(environmental, ("wind", "rainfall_mm", "elevation_m"))
        from src.agents.structural import calculate_structural_decision

        structural = calculate_structural_decision(environmental)

        return self.result(
            f"Load-bearing walls set to {structural.wall_thickness_mm}mm for regional resilience",
            structural.reasoning,
            structural.score,
        )


class SiteEngineerAgent(BaseAgent):
    name = "site_engineer"
    weight = 0.85

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        return self.result(
            f"Road-facing side set to {payload.plot.road_facing}",
            "Supports practical site access",
            7.8,
        )


class VastuExpertAgent(BaseAgent):
    name = "vastu_expert"
    weight = 0.7

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        if not payload.requirements.apply_vastu:
            return self.result(
                "Vastu optional adjustments skipped",
                "User disabled vastu preferences",
                7.0,
            )
        return self.result(
            "Kitchen placed in south-east zone",
            "Follows vastu guidance where practical",
            7.6,
        )


class InteriorDesignerAgent(BaseAgent):
    name = "interior_designer"
    weight = 0.75

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        return self.result(
            "Circulation path minimized across common rooms",
            "Improves comfort and usable space",
            8.1,
        )


class ConstructionBuilderAgent(BaseAgent):
    name = "construction_builder"
    weight = 0.9

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        return self.result(
            "Material schedule generated with climate-adaptive specs",
            "Construction-ready deliverable generated",
            8.3,
        )


class GraphExecutionError(RuntimeError):
    """Raised when the LangGraph design workflow fails to execute."""


class OrchestratorAgent:
    """Coordinates specialized agents via a LangGraph StateGraph workflow."""

    def execute(self, payload: AnalyzePlotRequest, environmental: dict) -> list[DesignDecision]:
        initial_state = {
            "payload": payload,
            "environmental": environmental,
            "agent_results": [],
            "decisions": [],
        }
        try:
            final_state = design_graph.invoke(initial_state)
        except Exception as exc:
            logger.exception("LangGraph design workflow execution failed")
            raise GraphExecutionError("Design workflow failed to execute") from exc
        return final_state["decisions"]
