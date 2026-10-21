"""Paddle planner, batched: sim/paddle_motion.py for thousands of worlds at once (torch, on the GPU).

The same rule as the CPU sim: the commanded point heads for its goal within a top speed and an
acceleration limit, and brakes so it stops on the goal. Run once per physics tick.

Usage:
    planner = BatchedPlanner(num_envs, low, high, max_speed, max_accel, device)
    planner.reset(env_ids, positions)
    planner.advance(goals, dt)          # goals (num_envs, 2)
    planner.position                    # (num_envs, 2) where each paddle's spring pulls it
"""

from __future__ import annotations

import torch

CLOSE_ENOUGH_M = 1e-4


class BatchedPlanner:
    def __init__(self, num_envs: int, low, high, max_speed: float, max_accel: float, device: str):
        self.low = torch.as_tensor(low, dtype=torch.float32, device=device)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=device)
        self.max_speed, self.max_accel = max_speed, max_accel
        self.position = torch.zeros(num_envs, 2, device=device)
        self.velocity = torch.zeros(num_envs, 2, device=device)

    def reset(self, env_ids, positions: torch.Tensor) -> None:
        self.position[env_ids] = positions
        self.velocity[env_ids] = 0.0

    def advance(self, goals: torch.Tensor, dt: float) -> None:
        goals = torch.maximum(torch.minimum(goals, self.high), self.low)
        to_goal = goals - self.position
        distance = to_goal.norm(dim=-1, keepdim=True)
        arrived = (distance < CLOSE_ENOUGH_M) & (
            self.velocity.norm(dim=-1, keepdim=True) < self.max_accel * dt
        )

        # 1. fastest speed that still lets us brake to a stop at the goal
        speed = torch.clamp(torch.sqrt(2 * self.max_accel * distance), max=self.max_speed)
        wanted = to_goal / distance.clamp(min=CLOSE_ENOUGH_M) * speed
        # 2. accelerate toward it, within the limit
        change = wanted - self.velocity
        size = change.norm(dim=-1, keepdim=True)
        change = change * torch.clamp(self.max_accel * dt / size.clamp(min=1e-12), max=1.0)
        velocity = self.velocity + change
        # 3. move, and stop dead against a limit
        moved = self.position + velocity * dt
        position = torch.maximum(torch.minimum(moved, self.high), self.low)
        velocity = torch.where(position == moved, velocity, torch.zeros_like(velocity))

        self.position = torch.where(arrived, goals, position)
        self.velocity = torch.where(arrived, torch.zeros_like(velocity), velocity)
