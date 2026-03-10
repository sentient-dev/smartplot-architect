"""LangGraph-based agentic workflow for multi-agent design orchestration."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.models.schemas import AnalyzePlotRequest, DesignDecision

ELEVATION_LOW_THRESHOLD = 150.0
ELEVATION_MID_THRESHOLD = 600.0


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentResult(TypedDict):
    name: str
    decision: str
    reasoning: str
    score: float
    weight: float


def _append_result(existing: list[AgentResult], new: list[AgentResult]) -> list[AgentResult]:
    """Reducer that appends new agent results to the accumulated list."""
    return existing + new


class DesignGraphState(TypedDict):
    """Shared state passed through every node of the design workflow graph."""

    payload: AnalyzePlotRequest
    environmental: dict
    agent_results: Annotated[list[AgentResult], _append_result]
    decisions: list[DesignDecision]


# ---------------------------------------------------------------------------
# Agent node helpers
# ---------------------------------------------------------------------------

def _make_result(name: str, decision: str, reasoning: str, score: float, weight: float) -> AgentResult:
    return AgentResult(name=name, decision=decision, reasoning=reasoning, score=score, weight=weight)


# ---------------------------------------------------------------------------
# Agent nodes — each reads from state and appends its AgentResult
# ---------------------------------------------------------------------------

def architect_node(state: DesignGraphState) -> dict:
    preferred = state["environmental"]["solar"]["preferred_exposure"]
    return {
        "agent_results": [
            _make_result(
                "architect",
                f"Primary living spaces aligned to {preferred}",
                "Optimized for natural daylight",
                8.5,
                1.0,
            )
        ]
    }


def meteorologist_node(state: DesignGraphState) -> dict:
    environmental = state["environmental"]
    if "wind" not in environmental:
        raise KeyError("Missing environmental keys for meteorologist: wind")
    wind = environmental["wind"]
    if "prevailing_direction" not in wind:
        raise KeyError("Missing environmental keys for meteorologist: wind.prevailing_direction")
    direction = wind["prevailing_direction"]
    return {
        "agent_results": [
            _make_result(
                "meteorologist",
                f"Cross-ventilation windows oriented towards {direction}",
                "Uses prevailing wind data",
                8.2,
                0.9,
            )
        ]
    }


def geologist_foundation_guidance(elevation: float) -> tuple[str, str]:
    if elevation < ELEVATION_LOW_THRESHOLD:
        return (
            f"Raised plinth foundation for low elevation site ({elevation}m)",
            "Low-lying terrain needs moisture and settlement safeguards",
        )
    if elevation < ELEVATION_MID_THRESHOLD:
        return (
            f"Reinforced strip footing for mid-elevation site ({elevation}m)",
            "Balanced soil pressure and drainage profile support standard reinforcement",
        )
    return (
        f"Stepped reinforced foundation for high elevation site ({elevation}m)",
        "Steeper terrain needs terrace-adaptive foundation stability",
    )


def geologist_node(state: DesignGraphState) -> dict:
    if "elevation_m" not in state["environmental"]:
        raise KeyError("Missing environmental keys for geologist: elevation_m")
    elevation = state["environmental"]["elevation_m"]
    decision, reasoning = geologist_foundation_guidance(elevation)
    return {
        "agent_results": [
            _make_result(
                "geologist",
                decision,
                reasoning,
                8.0,
                0.95,
            )
        ]
    }


def structural_engineer_node(state: DesignGraphState) -> dict:
    return {
        "agent_results": [
            _make_result(
                "structural_engineer",
                "Load-bearing walls increased to 230mm",
                "Safety-first structure for regional conditions",
                8.8,
                1.0,
            )
        ]
    }


def site_engineer_node(state: DesignGraphState) -> dict:
    road_facing = state["payload"].plot.road_facing
    return {
        "agent_results": [
            _make_result(
                "site_engineer",
                f"Road-facing side set to {road_facing}",
                "Supports practical site access",
                7.8,
                0.85,
            )
        ]
    }


def vastu_expert_node(state: DesignGraphState) -> dict:
    if not state["payload"].requirements.apply_vastu:
        return {
            "agent_results": [
                _make_result(
                    "vastu_expert",
                    "Vastu optional adjustments skipped",
                    "User disabled vastu preferences",
                    7.0,
                    0.7,
                )
            ]
        }
    return {
        "agent_results": [
            _make_result(
                "vastu_expert",
                "Kitchen placed in south-east zone",
                "Follows vastu guidance where practical",
                7.6,
                0.7,
            )
        ]
    }


def interior_designer_node(state: DesignGraphState) -> dict:
    return {
        "agent_results": [
            _make_result(
                "interior_designer",
                "Circulation path minimized across common rooms",
                "Improves comfort and usable space",
                8.1,
                0.75,
            )
        ]
    }


def construction_builder_node(state: DesignGraphState) -> dict:
    return {
        "agent_results": [
            _make_result(
                "construction_builder",
                "Material schedule generated with climate-adaptive specs",
                "Construction-ready deliverable generated",
                8.3,
                0.9,
            )
        ]
    }


def compile_decisions_node(state: DesignGraphState) -> dict:
    """Sort accumulated agent results by weighted score and build DesignDecision list."""
    ordered = sorted(
        state["agent_results"],
        key=lambda r: r["score"] * r["weight"],
        reverse=True,
    )
    decisions = [
        DesignDecision(
            agent=r["name"],
            decision=r["decision"],
            reasoning=r["reasoning"],
            score=round(r["score"] * r["weight"], 2),
        )
        for r in ordered
    ]
    return {"decisions": decisions}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_design_graph() -> CompiledStateGraph:
    """Construct and compile the LangGraph design workflow."""
    workflow = StateGraph(DesignGraphState)

    # Register specialist agent nodes
    workflow.add_node("architect", architect_node)
    workflow.add_node("meteorologist", meteorologist_node)
    workflow.add_node("geologist", geologist_node)
    workflow.add_node("structural_engineer", structural_engineer_node)
    workflow.add_node("site_engineer", site_engineer_node)
    workflow.add_node("vastu_expert", vastu_expert_node)
    workflow.add_node("interior_designer", interior_designer_node)
    workflow.add_node("construction_builder", construction_builder_node)
    workflow.add_node("compile_decisions", compile_decisions_node)

    # Sequential pipeline: each specialist feeds into the next
    workflow.set_entry_point("architect")
    workflow.add_edge("architect", "meteorologist")
    workflow.add_edge("meteorologist", "geologist")
    workflow.add_edge("geologist", "structural_engineer")
    workflow.add_edge("structural_engineer", "site_engineer")
    workflow.add_edge("site_engineer", "vastu_expert")
    workflow.add_edge("vastu_expert", "interior_designer")
    workflow.add_edge("interior_designer", "construction_builder")
    workflow.add_edge("construction_builder", "compile_decisions")
    workflow.add_edge("compile_decisions", END)

    return workflow.compile()


# Module-level compiled graph (singleton)
try:
    design_graph: CompiledStateGraph = build_design_graph()
except Exception as exc:  # pragma: no cover - defensive initialization guard
    raise RuntimeError("Failed to build design graph during module import") from exc
