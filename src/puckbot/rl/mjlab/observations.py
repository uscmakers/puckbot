"""Observation terms: what the policy sees (the same layout as the CPU training env).

    camera frames   the pucks and the opponent paddles in the newest 120 fps frame and older ones
                    (latency, frame timing and lags from the camera sensor, camera.py; noise added here)
    own paddle      our paddle's x, y and vx, vy (what the ESP32 reports)
    last action     (dx, dy) from the previous tick           -> mjlab's mdp.last_action

Everything is scaled to about [-1, 1]: positions by the table's half size, speeds by the top speed.

Usage (in the env config):
    ObservationTermCfg(func=camera_frames, params={"facts": facts, "noise_m": 0.0015})
    ObservationTermCfg(func=OwnPaddle, params={"facts": facts})
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from puckbot.rl.mjlab.scene import CAMERA, TableFacts
from puckbot.rl.mjlab.state import TableState

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _half_size(facts: TableFacts, device) -> torch.Tensor:
    return torch.tensor([facts.geometry.rail_x, facts.geometry.rail_y], device=device)


def camera_frames(env: ManagerBasedRlEnv, facts: TableFacts, noise_m: float) -> torch.Tensor:
    """Noisy positions in the camera frames, scaled. Shape (num_envs, lags * objects * 2)."""
    frames = env.scene[CAMERA].data
    frames = frames + noise_m * torch.randn_like(frames)
    return (frames / _half_size(facts, env.device)).flatten(start_dim=1)


class OwnPaddle:
    """Our paddle (paddle 0): x, y scaled by the table's half size; vx, vy scaled by the top speed."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        facts: TableFacts = cfg.params["facts"]
        self.state = TableState(env, facts)
        self.half_size = _half_size(facts, env.device)
        self.top_speed = facts.settings.paddle.max_speed_mps

    def __call__(self, env: ManagerBasedRlEnv, facts: TableFacts) -> torch.Tensor:
        snap = self.state.read()
        return torch.cat(
            [snap.paddle_pos[:, 0] / self.half_size, snap.paddle_vel[:, 0] / self.top_speed], dim=-1
        )
