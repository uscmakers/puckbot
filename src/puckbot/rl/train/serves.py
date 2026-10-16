"""Serves: the random shots the goalie has to stop.

Each serve picks a puck start, a speed and an aim:
    - start     anywhere on the far half of the table
    - aim       a point on our goal line, a bit wider than the slot (some shots miss, so doing
                nothing is not always wrong, and the goalie must learn to tell)
    - bank      with some probability the aim is the goal's mirror image behind a side rail,
                so the puck bounces once first (rail energy loss is ignored; it is only an aim)

Slow pucks first (Hudson). The sim's puck sometimes gains speed near the goal mouths above
~4 m/s, so the default range stays below that.

Usage:
    serve = random_serve(random, geometry, settings.goalie)
    env.reset(puck_pos=serve.puck_pos, puck_vel=serve.puck_vel)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puckbot.rl.sim.model import TableGeometry
from puckbot.rl.train.settings import GoalieSettings

AIM_SPREAD_M = 0.10  # aim points go this far beyond the goal slot's edges
START_X_FRACTION = (0.10, 0.90)  # the puck starts this far toward the far end (0 = centre line, 1 = far rail)


@dataclass(frozen=True)
class Serve:
    puck_pos: np.ndarray
    puck_vel: np.ndarray
    banks: bool


def random_serve(random: np.random.Generator, geometry: TableGeometry, goalie: GoalieSettings) -> Serve:
    limit_y = geometry.puck_limit_y
    start = np.array(
        [
            random.uniform(*START_X_FRACTION) * geometry.rail_x,
            random.uniform(-0.8, 0.8) * limit_y,
        ]
    )
    aim_y = random.uniform(-1, 1) * (geometry.goal_half_width + AIM_SPREAD_M)
    banks = bool(random.random() < goalie.bank_shot_fraction)
    if banks:
        wall_y = np.sign(start[1] or 1.0) * limit_y
        aim_y = 2 * wall_y - aim_y  # mirror image of the aim point behind that rail
    aim = np.array([-geometry.rail_x, aim_y])

    direction = (aim - start) / np.linalg.norm(aim - start)
    speed = random.uniform(*goalie.puck_speed_mps)
    return Serve(puck_pos=start, puck_vel=direction * speed, banks=banks)
