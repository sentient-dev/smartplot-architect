"""Shared deterministic construction-builder logic."""

from __future__ import annotations

from src.models.schemas import AnalyzePlotRequest

DEFAULT_CONSTRUCTION_SCORE = 8.3
RAINFALL_THRESHOLD_HEAVY = 1000
RAINFALL_THRESHOLD_MODERATE = 800
TEMP_THRESHOLD_HIGH_ALBEDO = 25

_PROCUREMENT_BY_BUDGET = {
    "premium": "BIM quantity takeoff with phased premium procurement",
    "mid-range": "batch quantity takeoff with regional supplier sequencing",
    "economy": "value-engineered quantity takeoff with local supplier sequencing",
    "low": "value-engineered quantity takeoff with local supplier sequencing",
    "value-engineered": "value-engineered quantity takeoff with local supplier sequencing",
}


def generate_construction_builder_output(payload: AnalyzePlotRequest, environmental: dict) -> tuple[str, str, float]:
    try:
        rainfall = float(environmental["rainfall_mm"])
    except KeyError as exc:
        raise KeyError("Missing required environmental key 'rainfall_mm' for construction builder output.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid environmental value for 'rainfall_mm'; expected a numeric value.") from exc

    try:
        avg_temp = float(environmental["weather"]["average_temp_c"])
    except KeyError as exc:
        raise KeyError(
            "Missing required environmental key 'weather.average_temp_c' for construction builder output."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid environmental value for 'weather.average_temp_c'; expected a numeric value."
        ) from exc
    budget = payload.requirements.budget.lower()

    if rainfall >= RAINFALL_THRESHOLD_HEAVY:
        envelope = "reinforced concrete + membrane waterproofing"
    elif rainfall >= RAINFALL_THRESHOLD_MODERATE:
        envelope = "damp-proofed masonry + anti-corrosion steel"
    else:
        envelope = "thermally rendered masonry"

    thermal_spec = "high-albedo insulated roof" if avg_temp >= TEMP_THRESHOLD_HIGH_ALBEDO else "standard insulated roof"
    procurement = _PROCUREMENT_BY_BUDGET.get(
        budget,
        "value-engineered quantity takeoff with local supplier sequencing",
    )
    decision = f"Construction package: {envelope}; {thermal_spec}; schedule: {procurement}"
    reasoning = "Climate-adaptive material scheduling produced deterministic construction-ready specifications"
    return decision, reasoning, DEFAULT_CONSTRUCTION_SCORE
