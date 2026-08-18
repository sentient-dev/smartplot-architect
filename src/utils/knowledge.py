"""Load seed knowledge-base files shipped with the repository."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_vastu_rules() -> dict[str, str]:
    return _read_json(DATA_DIR / "vastu_rules.json")


@lru_cache(maxsize=1)
def load_material_specs() -> dict[str, dict[str, Any]]:
    return _read_json(DATA_DIR / "material_specs.json")


@lru_cache(maxsize=1)
def load_ibc_minimums() -> dict[str, float]:
    return _read_json(DATA_DIR / "building_codes" / "ibc_minimums.json")
