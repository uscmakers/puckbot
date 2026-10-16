"""Paddle motion: turn "go to (x, y)" into a move the gantry can really make.

The real gantry has a top speed and a limit on how hard it can accelerate. This
mimics both: it moves a *commanded point* toward the goal, and the paddle follows it.

Steps (every physics tick):
    1. Pick the fastest safe speed toward the goal (it must be able to brake in time)
    2. Change the commanded velocity toward it, but by no more than the accel limit
    3. Move the commanded point, and keep it inside the paddle limits

Usage:
    motion = PaddleMotion(settings.paddle, low, high, start=(-0.85, 0))
    motion.advance(goal=(-0.5, 0.2), dt=0.001)
    motion.position     # where the paddle should be right now
"""

from __future__ import annotations

import numpy as np

from puckbot.sim.settings import PaddleSettings

CLOSE_ENOUGH_M = 1e-4


class PaddleMotion:
    def __init__(self, paddle: PaddleSettings, low, high, start):
        self.max_speed = paddle.max_speed_mps
        self.max_accel = paddle.max_accel_mps2
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.position = np.asarray(start, dtype=float)
        self.velocity = np.zeros(2)

    def reset(self, position) -> None:
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.zeros(2)

    def advance(self, goal, dt: float) -> None:
        goal = np.clip(np.asarray(goal, dtype=float), self.low, self.high)
        to_goal = goal - self.position
        distance = np.linalg.norm(to_goal)

        if distance < CLOSE_ENOUGH_M and np.linalg.norm(self.velocity) < self.max_accel * dt:
            self.position, self.velocity = goal, np.zeros(2)
            return

        # 1. fastest speed that still lets us brake to a stop at the goal
        brake_speed = np.sqrt(2 * self.max_accel * distance)
        wanted = to_goal / max(distance, CLOSE_ENOUGH_M) * min(self.max_speed, brake_speed)

        # 2. accelerate toward it, within the limit
        change = wanted - self.velocity
        biggest_change = self.max_accel * dt
        size = np.linalg.norm(change)
        if size > biggest_change:
            change *= biggest_change / size
        self.velocity = self.velocity + change

        # 3. move, and stop dead against a limit
        moved = self.position + self.velocity * dt
        self.position = np.clip(moved, self.low, self.high)
        self.velocity = np.where(self.position == moved, self.velocity, 0.0)
