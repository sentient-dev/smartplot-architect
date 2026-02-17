"""Lightweight multi-agent orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.schemas import AnalyzePlotRequest, DesignDecision


@dataclass(frozen=True)
class AgentResult:
    name: str
    decision: str
    reasoning: str
    score: float
    weight: float


class BaseAgent:
    name = "base"
    weight = 0.5

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:  # pragma: no cover - abstract
        raise NotImplementedError


class ArchitectAgent(BaseAgent):
    name = "architect"
    weight = 1.0

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        preferred = environmental["solar"]["preferred_exposure"]
        return AgentResult(self.name, f"Primary living spaces aligned to {preferred}", "Optimized for natural daylight", 8.5, self.weight)


class MeteorologistAgent(BaseAgent):
    name = "meteorologist"
    weight = 0.9

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        direction = environmental["wind"]["prevailing_direction"]
        return AgentResult(self.name, f"Cross-ventilation windows oriented towards {direction}", "Uses prevailing wind data", 8.2, self.weight)


class GeologistAgent(BaseAgent):
    name = "geologist"
    weight = 0.95

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        elevation = environmental["elevation_m"]
        return AgentResult(self.name, f"Foundation tuned for elevation {elevation}m", "Reduced moisture and settlement risks", 8.0, self.weight)


class StructuralEngineerAgent(BaseAgent):
    name = "structural_engineer"
    weight = 1.0

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        return AgentResult(self.name, "Load-bearing walls increased to 230mm", "Safety-first structure for regional conditions", 8.8, self.weight)


class SiteEngineerAgent(BaseAgent):
    name = "site_engineer"
    weight = 0.85

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        return AgentResult(self.name, f"Road-facing side set to {payload.plot.road_facing}", "Supports practical site access", 7.8, self.weight)


class VastuExpertAgent(BaseAgent):
    name = "vastu_expert"
    weight = 0.7

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        if not payload.requirements.apply_vastu:
            return AgentResult(self.name, "Vastu optional adjustments skipped", "User disabled vastu preferences", 7.0, self.weight)
        return AgentResult(self.name, "Kitchen placed in south-east zone", "Follows vastu guidance where practical", 7.6, self.weight)


class InteriorDesignerAgent(BaseAgent):
    name = "interior_designer"
    weight = 0.75

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        return AgentResult(self.name, "Circulation path minimized across common rooms", "Improves comfort and usable space", 8.1, self.weight)


class ConstructionBuilderAgent(BaseAgent):
    name = "construction_builder"
    weight = 0.9

    def run(self, payload: AnalyzePlotRequest, environmental: dict) -> AgentResult:
        return AgentResult(self.name, "Material schedule generated with climate-adaptive specs", "Construction-ready deliverable generated", 8.3, self.weight)


class OrchestratorAgent:
    """Coordinates specialized agents with weighted conflict resolution."""

    def __init__(self) -> None:
        self._agents: list[BaseAgent] = [
            ArchitectAgent(),
            MeteorologistAgent(),
            GeologistAgent(),
            StructuralEngineerAgent(),
            SiteEngineerAgent(),
            VastuExpertAgent(),
            InteriorDesignerAgent(),
            ConstructionBuilderAgent(),
        ]

    def execute(self, payload: AnalyzePlotRequest, environmental: dict) -> list[DesignDecision]:
        results = [agent.run(payload, environmental) for agent in self._agents]
        ordered = sorted(results, key=lambda item: item.score * item.weight, reverse=True)
        return [
            DesignDecision(
                agent=result.name,
                decision=result.decision,
                reasoning=result.reasoning,
                score=round(result.score * result.weight, 2),
            )
            for result in ordered
        ]
