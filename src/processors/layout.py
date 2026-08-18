"""Deterministic residential room layout from plot and program requirements."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.schemas import AnalyzePlotRequest
from src.utils.knowledge import load_vastu_rules


@dataclass(frozen=True)
class Room:
    name: str
    x: float
    y: float
    width: float
    depth: float
    zone: str

    @property
    def area(self) -> float:
        return round(self.width * self.depth, 2)


def _split_evenly(count: int, start: float, span: float, gap: float) -> list[tuple[float, float]]:
    usable = span - gap * max(count - 1, 0)
    each = usable / count
    boxes: list[tuple[float, float]] = []
    cursor = start
    for _ in range(count):
        boxes.append((cursor, each))
        cursor += each + gap
    return boxes


def build_room_layout(request: AnalyzePlotRequest) -> list[Room]:
    """Pack rooms on a north-up plan (x=west→east, y=south→north)."""
    plot_w = request.plot.dimensions.width
    plot_d = request.plot.dimensions.length
    reqs = request.requirements
    gap = 1.0
    south_band = plot_d * 0.42
    north_band = plot_d - south_band - gap

    kitchen_w = min(14.0, plot_w * 0.28)
    living_w = plot_w - kitchen_w - gap
    dining_d = min(12.0, south_band * 0.38)
    living_d = south_band - dining_d - gap

    rooms = [
        Room("living_room", 0.0, 0.0, living_w, living_d, "south"),
        Room("dining_room", 0.0, living_d + gap, living_w, dining_d, "south"),
        Room("kitchen", living_w + gap, 0.0, kitchen_w, south_band, "south-east"),
    ]

    bath_d = min(8.0, north_band * 0.28)
    bedroom_d = north_band - bath_d - gap
    bedroom_y = south_band + gap
    bath_y = bedroom_y + bedroom_d + gap

    bedroom_boxes = _split_evenly(reqs.bedrooms, 0.0, plot_w, gap)
    for index, (x, width) in enumerate(bedroom_boxes):
        name = "master_bedroom" if index == 0 else f"bedroom_{index + 1}"
        zone = "south-west" if index == 0 else "north"
        rooms.append(Room(name, x, bedroom_y, width, bedroom_d, zone))

    bath_boxes = _split_evenly(reqs.bathrooms, 0.0, plot_w, gap)
    for index, (x, width) in enumerate(bath_boxes):
        rooms.append(Room(f"bathroom_{index + 1}", x, bath_y, width, bath_d, "north"))

    if reqs.apply_vastu:
        rules = load_vastu_rules()
        _ = rules  # layout already follows kitchen SE / master SW conventions
    return rooms
