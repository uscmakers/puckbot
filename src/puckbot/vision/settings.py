"""Settings: what the table, puck and mallets look like.

Values live in configs/our_table.json. This file says what each one means.

Colours are HSV ranges (OpenCV scale):
    H  hue         which colour       0-179
    S  saturation  how colourful      0-255
    V  value       how bright         0-255
"""

from __future__ import annotations

import json
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import get_type_hints

DEFAULT_SETTINGS = Path(__file__).resolve().parent / "configs" / "our_table.json"

HSV = tuple[int, int, int]


@dataclass
class TableSettings:
    outer_width_in: float  # whole table, outside edge to outside edge
    outer_length_in: float
    rail_width_in: float  # rail on each side
    surface_hsv_lower: HSV  # colour of the playing surface (blue)
    surface_hsv_upper: HSV

    @property
    def surface_size_in(self) -> tuple[float, float]:
        """Playing surface (width, length) = outer size minus a rail on each side."""
        rails = 2 * self.rail_width_in
        return self.outer_width_in - rails, self.outer_length_in - rails


@dataclass
class ShapeRules:
    """What a puck or mallet looks like in the photo. Sizes in pixels."""

    expected_radius_px: float  # typical size; the closest match ranks first
    min_area_px: float
    max_area_px: float
    min_roundness: float  # 1.0 = perfect circle
    min_fill: float  # 1.0 = solid disc; a ring scores low


@dataclass
class DetectorSettings:
    white_hsv_lower: HSV  # colour of the puck and mallets
    white_hsv_upper: HSV
    puck: ShapeRules
    mallet: ShapeRules
    blur_px: int = 5  # smooths camera noise before the colour test
    cleanup_px: int = 5  # removes specks and fills small holes after it


@dataclass
class Settings:
    table: TableSettings
    detector: DetectorSettings


def load_settings(path: str | Path = DEFAULT_SETTINGS) -> Settings:
    """Read a settings file. A typo'd or missing key raises an error."""
    return _from_dict(Settings, json.loads(Path(path).read_text()))


def _from_dict(cls, data: dict):
    """JSON dict -> dataclass. Nested dicts become nested dataclasses, lists become tuples."""
    types = get_type_hints(cls)
    unknown = set(data) - set(types)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown settings {sorted(unknown)}")

    values = {}
    for name, value in data.items():
        if is_dataclass(types[name]):
            value = _from_dict(types[name], value)
        elif isinstance(value, list):
            value = tuple(value)
        values[name] = value
    return cls(**values)
