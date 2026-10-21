"""Camera, batched: train/camera.py as an mjlab sensor, for every world at once.

The real camera takes a frame every 1/120 s; a frame reaches the policy `latency_s` later. The
policy acts at 50 Hz, so each tick it sees the newest frame that has arrived, plus a few older
frames (the history lags, counted in camera frames). The sensor records the tracked objects (the
pucks, then the opponent paddles) after every physics tick, so frames land on the right tick.

Usage (scene + observation):
    PuckCameraCfg(name="camera", facts=facts, camera=train_settings.camera)
    env.scene["camera"].data            # (num_envs, lags, objects, 2) metres, newest frame first
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
from mjlab.sensor.sensor import Sensor, SensorCfg

from puckbot.rl.mjlab.scene import ENTITY, TableFacts
from puckbot.rl.train.settings import CameraSettings


def frame_ticks(tick: torch.Tensor, physics_dt: float, camera: CameraSettings) -> torch.Tensor:
    """For each world: the physics tick of the newest arrived frame and of each older lag frame.

    `tick` = physics ticks since the episode started (tick 0 is the start). Returns (worlds, lags).
    Same rule as train/camera.py CameraModel.latest_frames.
    """
    frame_s = 1.0 / camera.rate_hz
    lags = torch.as_tensor(camera.history_lags_frames, device=tick.device)
    newest = torch.floor((tick.double() * physics_dt - camera.latency_s) / frame_s)
    times = (newest.unsqueeze(-1) - lags) * frame_s
    ticks = torch.round(times / physics_dt).long()
    return torch.minimum(ticks.clamp(min=0), tick.unsqueeze(-1))


@dataclass
class PuckCameraCfg(SensorCfg):
    facts: TableFacts = None  # type: ignore[assignment]
    camera: CameraSettings = None  # type: ignore[assignment]

    def build(self) -> PuckCamera:
        return PuckCamera(self)


class PuckCamera(Sensor[torch.Tensor]):
    def __init__(self, cfg: PuckCameraCfg):
        super().__init__()
        self.cfg = cfg
        facts = cfg.facts
        self._joints = [f"{ENTITY}/puck_{p}_{axis}" for p in range(facts.n_pucks) for axis in "xy"]
        self._joints += [f"{ENTITY}/paddle_{k}_{axis}" for k in range(1, facts.n_paddles) for axis in "xy"]
        oldest_s = cfg.camera.latency_s + (max(cfg.camera.history_lags_frames) + 2) / cfg.camera.rate_hz
        self._kept = int(oldest_s / facts.physics_dt) + 2  # ring buffer length, in physics ticks
        self._physics_dt = facts.physics_dt

    def edit_spec(self, scene_spec: mujoco.MjSpec, entities) -> None:
        pass  # reads joint positions; adds nothing to the model

    def initialize(self, mj_model, model, data, device: str) -> None:
        self._data = data
        self._qpos_ids = torch.tensor(
            [int(mj_model.joint(n).qposadr[0]) for n in self._joints], device=device
        )
        worlds = data.qpos.shape[0]
        self._buffer = torch.zeros(worlds, self._kept, len(self._joints), device=device)
        self._tick = torch.zeros(worlds, dtype=torch.long, device=device)
        self._fresh = torch.ones(worlds, dtype=torch.bool, device=device)
        self._rows = torch.arange(worlds, device=device)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        env_ids = slice(None) if env_ids is None else env_ids
        self._tick[env_ids] = 0
        self._fresh[env_ids] = True  # filled on the next physics tick, once the reset events have run

    def update(self, dt: float) -> None:
        super().update(dt)
        positions = self._data.qpos[:, self._qpos_ids]
        self._buffer[self._fresh] = positions[self._fresh].unsqueeze(1)  # before the start: sitting still
        self._fresh[:] = False
        self._tick += 1
        self._buffer[self._rows, self._tick % self._kept] = positions

    def _compute_data(self) -> torch.Tensor:
        ticks = frame_ticks(self._tick, self._physics_dt, self.cfg.camera)  # (worlds, lags)
        frames = self._buffer[self._rows.unsqueeze(-1), ticks % self._kept]  # (worlds, lags, objects*2)
        return frames.reshape(frames.shape[0], frames.shape[1], -1, 2)
