"""Scientific validators for generated designs."""

from __future__ import annotations

from src.models.schemas import AnalyzePlotRequest, DesignDecision, ValidationReport


class ScientificValidator:
    """Applies rule-based checks to enforce science-first design quality."""

    def evaluate(self, request: AnalyzePlotRequest, environmental: dict, decisions: list[DesignDecision]) -> ValidationReport:
        preferred = environmental["solar"]["preferred_exposure"]
        orientation_match = 1.0 if request.plot.orientation.lower().startswith(preferred[0]) else 0.6

        ventilation_match = any("ventilation" in decision.decision.lower() for decision in decisions)
        ventilation_score = 8.5 if ventilation_match else 6.0

        structural_match = any("load-bearing" in decision.decision.lower() for decision in decisions)
        structural_score = 9.0 if structural_match else 6.5

        sunlight_score = round(8.0 * orientation_match, 2)
        avg = (sunlight_score + ventilation_score + structural_score) / 3

        issues: list[str] = []
        if sunlight_score < 7.0:
            issues.append("Plot orientation is not optimal for local sun path")
        if ventilation_score < 7.0:
            issues.append("Cross-ventilation recommendation missing")
        if structural_score < 7.0:
            issues.append("Structural load validation recommendation missing")

        return ValidationReport(
            sunlight_score=sunlight_score,
            ventilation_score=ventilation_score,
            structural_score=structural_score,
            energy_efficiency="A" if avg >= 8 else "B",
            compliant=avg >= 7.5,
            issues=issues,
        )
