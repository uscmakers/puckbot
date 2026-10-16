"""Settings: the numbers we can tune in the sim.

Values live in configs/sim.json. This file says what each one means.
Table, puck and paddle shapes are NOT here: they come from the CAD model (cad/table_urdf/).

Units: metres, seconds, newtons. Frame = the CAD frame:
    origin  centre of the table      x  along the table (robot defends -x)
    y       across the table         z  up
"""

from __future__ import annotations

import json
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import get_type_hints

DEFAULT_SETTINGS = Path(__file__).resolve().parent / "configs" / "sim.json"
DEFAULT_TABLE_MODEL = Path(__file__).resolve().parents[4] / "cad" / "table_urdf" / "table.xml"


@dataclass
class PaddleSettings:
    max_speed_mps: float  # how fast the target point may move: the mallet's top speed
    max_accel_mps2: float  # how hard the target point may speed up or brake
    response_time_s: float  # the mallet catches up with its target in about this long (smaller = stiffer)
    wall_margin_m: float  # paddle stays this far from the rails (the real soft limit)
    home_position_m: tuple[float, float]  # where the paddle waits (x, y)


@dataclass
class PuckSettings:
    slowdown_ns_per_m: float  # MuJoCo joint damping: a simple stand-in for friction + drag
    bounce_damping_ratio: (
        float  # rail bounce: 1.0 = dead (CAD default), lower = livelier. Below ~0.1 it blows up
    )


@dataclass
class Settings:
    control_rate_hz: float  # how often the controller picks a new target
    physics_timestep_s: float  # MuJoCo tick. Rails bounce stably only at 0.5 ms or smaller
    paddle: PaddleSettings
    puck: PuckSettings


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
