"""Paddle motion: the path the commanded point takes to its target, within a speed and acceleration limit.

The paddle follows this point through a MuJoCo spring (model.py), so all the physics (inertia,
collisions) is MuJoCo's. This is only the "how fast may the target move" rule: a top speed and an
acceleration limit, with braking planned so the point stops on its target instead of overshooting.
MuJoCo has no equivalent for that planning.

Steps (every physics tick):
    1. Pick the fastest safe speed toward the goal (it must be able to brake in time)
    2. Change the commanded velocity toward it, but by no more than the accel limit
    3. Move the commanded point, and keep it inside the paddle limits

Plain Python floats, not numpy: this runs 40 times per control step per paddle, and numpy's
overhead on 2-element arrays was most of the sim's run time.

Usage:
    motion = PaddleMotion(settings.paddle, low, high, start=(-0.85, 0))
    motion.advance(goal_x=-0.5, goal_y=0.2, dt=0.0005)
    motion.x, motion.y      # where the paddle should be right now
"""

from __future__ import annotations

from math import hypot, sqrt

import numpy as np

from puckbot.rl.sim.settings import PaddleSettings

CLOSE_ENOUGH_M = 1e-4


class PaddleMotion:
    def __init__(self, paddle: PaddleSettings, low, high, start):
        self.max_speed = paddle.max_speed_mps
        self.max_accel = paddle.max_accel_mps2
        self.low_x, self.low_y = (float(v) for v in low)
        self.high_x, self.high_y = (float(v) for v in high)
        self.reset(start)

    def reset(self, position) -> None:
        self.x, self.y = (float(v) for v in position)
        self.vx = self.vy = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    def advance(self, goal_x: float, goal_y: float, dt: float) -> None:
        goal_x = min(max(goal_x, self.low_x), self.high_x)
        goal_y = min(max(goal_y, self.low_y), self.high_y)
        to_goal_x, to_goal_y = goal_x - self.x, goal_y - self.y
        distance = hypot(to_goal_x, to_goal_y)

        if distance < CLOSE_ENOUGH_M and hypot(self.vx, self.vy) < self.max_accel * dt:
            self.x, self.y, self.vx, self.vy = goal_x, goal_y, 0.0, 0.0
            return

        # 1. fastest speed that still lets us brake to a stop at the goal
        speed = min(self.max_speed, sqrt(2 * self.max_accel * distance))
        scale = speed / max(distance, CLOSE_ENOUGH_M)

        # 2. accelerate toward it, within the limit
        change_x, change_y = to_goal_x * scale - self.vx, to_goal_y * scale - self.vy
        size, biggest_change = hypot(change_x, change_y), self.max_accel * dt
        if size > biggest_change:
            change_x, change_y = change_x * biggest_change / size, change_y * biggest_change / size
        self.vx += change_x
        self.vy += change_y

        # 3. move, and stop dead against a limit
        moved_x, moved_y = self.x + self.vx * dt, self.y + self.vy * dt
        self.x = min(max(moved_x, self.low_x), self.high_x)
        self.y = min(max(moved_y, self.low_y), self.high_y)
        if self.x != moved_x:
            self.vx = 0.0
        if self.y != moved_y:
            self.vy = 0.0
