"""Observation: the numbers the policy sees each control tick.

Layout (all scaled to about [-1, 1]), 16 numbers with one puck and one paddle:
    tracked objects' x, y in the newest camera frame and older ones   2 x lags x objects   (camera.py)
        objects = the pucks, then the other paddles (the opponent)
    our paddle x, y                                                    2   what the ESP32 reports
    our paddle vx, vy                                                  2
    previous action dx, dy                                             2

Raw positions, not velocities: right after a bounce a velocity estimate is wrong, positions are not
(Hudson's lesson). The policy can work out the velocity from the history.

Usage:
    encoder = ObservationEncoder(env.geometry, env.settings)
    vector = encoder.encode(frames, state, previous_action)
"""

from __future__ import annotations

import numpy as np

from puckbot.rl.sim.env import State
from puckbot.rl.sim.model import TableGeometry
from puckbot.rl.sim.settings import Settings


class ObservationEncoder:
    def __init__(self, geometry: TableGeometry, settings: Settings):
        self.half_size = np.array([geometry.rail_x, geometry.rail_y])
        self.max_speed = settings.paddle.max_speed_mps

    def size(self, lags: int, objects: int) -> int:
        return 2 * lags * objects + 6

    def encode(self, frames: np.ndarray, state: State, previous_action: np.ndarray) -> np.ndarray:
        """`frames`: (lags, objects, 2) from the camera. We are paddle 0."""
        return np.concatenate(
            [
                (frames / self.half_size).ravel(),
                state.paddle_pos[0] / self.half_size,
                state.paddle_vel[0] / self.max_speed,
                previous_action,
            ]
        ).astype(np.float32)
