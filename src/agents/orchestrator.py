"""Multi-agent orchestration layer powered by LangGraph."""

from __future__ import annotations

import logging

from src.agents.graph import design_graph
from src.models.schemas import AnalyzePlotRequest, DesignDecision

logger = logging.getLogger(__name__)


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
