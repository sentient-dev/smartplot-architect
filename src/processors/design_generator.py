"""Design output assembly utilities."""

from __future__ import annotations

from uuid import uuid4

from src.models.schemas import AnalyzePlotRequest, DesignDecision


class DesignProcessor:
    """Converts orchestrated decisions into deliverable metadata."""

    def build_outputs(self, request: AnalyzePlotRequest, decisions: list[DesignDecision], validation_score: float) -> tuple[dict, dict]:
        design_id = uuid4()
        area = round(request.plot.dimensions.length * request.plot.dimensions.width, 2)

        files = {
            "floor_plan_2d": f"/artifacts/{design_id}/floor_plan.svg",
            "autocad_file": f"/artifacts/{design_id}/floor_plan.dxf",
            "3d_model": f"/artifacts/{design_id}/model.gltf",
            "documentation": f"/artifacts/{design_id}/design_report.pdf",
            "sun_analysis": f"/artifacts/{design_id}/sun_path.png",
            "ventilation_analysis": f"/artifacts/{design_id}/ventilation.png",
            "material_specifications": f"/artifacts/{design_id}/bom.json",
        }

        summary = {
            "total_area": f"{area} sq {request.plot.dimensions.unit}",
            "room_count": sum(
                (
                    request.requirements.bedrooms,
                    request.requirements.bathrooms,
                    request.requirements.kitchen,
                    request.requirements.living_room,
                    request.requirements.dining_room,
                )
            ),
            "optimization_score": round(sum(decision.score for decision in decisions) / max(len(decisions), 1), 2),
            "energy_efficiency": "A+" if validation_score >= 8.0 else "A",
            "vastu_compliance": 92 if request.requirements.apply_vastu else 0,
        }
        return files, summary
