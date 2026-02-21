"""Multi-agent orchestration layer powered by LangGraph."""

from __future__ import annotations

from src.agents.graph import design_graph
from src.models.schemas import AnalyzePlotRequest, DesignDecision


class OrchestratorAgent:
    """Coordinates specialized agents via a LangGraph StateGraph workflow."""

    def execute(self, payload: AnalyzePlotRequest, environmental: dict) -> list[DesignDecision]:
        initial_state = {
            "payload": payload,
            "environmental": environmental,
            "agent_results": [],
            "decisions": [],
        }
        final_state = design_graph.invoke(initial_state)
        return final_state["decisions"]
