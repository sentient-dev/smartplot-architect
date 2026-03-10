"""Shared structural engineering decision logic."""

from __future__ import annotations

from dataclasses import dataclass

# Safety-first thresholds tuned for deterministic regional recommendations.
MIN_RAINFALL_FOR_250MM = 1200.0
MIN_RAINFALL_FOR_300MM = 1600.0
MIN_WIND_FOR_250MM = 6.0
MIN_WIND_FOR_300MM = 7.5
MIN_ELEVATION_FOR_300MM = 600.0


@dataclass(frozen=True)
class StructuralDecision:
    wall_thickness_mm: int
    reasoning: str
    score: float


def calculate_structural_decision(environmental: dict) -> StructuralDecision:
    try:
        wind_speed = float(environmental.get("wind", {}).get("avg_speed_mps", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid environmental value for 'wind.avg_speed_mps'; expected a numeric value.") from exc
    try:
        rainfall = float(environmental.get("rainfall_mm", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid environmental value for 'rainfall_mm'; expected a numeric value.") from exc
    try:
        elevation = float(environmental.get("elevation_m", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid environmental value for 'elevation_m'; expected a numeric value.") from exc

    wall_thickness_mm = 230
    if rainfall >= MIN_RAINFALL_FOR_250MM or wind_speed >= MIN_WIND_FOR_250MM:
        wall_thickness_mm = 250
    if (
        rainfall >= MIN_RAINFALL_FOR_300MM
        or wind_speed >= MIN_WIND_FOR_300MM
        or elevation >= MIN_ELEVATION_FOR_300MM
    ):
        wall_thickness_mm = 300

    score = 8.6
    if wall_thickness_mm == 250:
        score = 8.9
    elif wall_thickness_mm == 300:
        score = 9.1

    return StructuralDecision(
        wall_thickness_mm=wall_thickness_mm,
        reasoning=(
            f"Safety-first structure using wind {wind_speed:.2f}m/s, "
            f"rainfall {rainfall:.2f}mm, elevation {elevation:.2f}m"
        ),
        score=score,
    )
