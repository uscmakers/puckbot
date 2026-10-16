"""Camera model: what the policy would really see of the puck (and the opponent's paddle).

The real camera takes a picture every 1/120 s. The picture reaches the policy a bit later (latency),
and each position in it is slightly off (noise). The policy runs at its own rate (50 Hz), so at each
tick it just takes the newest picture that has arrived.

Steps:
    1. Record   remember where every tracked object was after every physics tick
    2. Arrive   at control time T, the newest frame is the last one taken before T - latency
    3. Look     report the objects in that frame and in a few older ones (the history lags), with noise

The tracked objects are whatever the caller records: here the pucks, then the other paddles.

Usage:
    camera = CameraModel(settings.camera, objects=2, physics_dt=0.0005, random=np.random.default_rng(0))
    camera.reset(positions)                 # (objects, 2) at the start of an episode
    camera.record(trace)                    # (physics ticks, objects, 2), after every env.step
    frames = camera.latest_frames()         # (number of lags, objects, 2) metres, newest first
"""

from __future__ import annotations

import numpy as np

from puckbot.rl.train.settings import CameraSettings


class CameraModel:
    def __init__(self, camera: CameraSettings, objects: int, physics_dt: float, random: np.random.Generator):
        self.camera = camera
        self.physics_dt = physics_dt
        self.frame_seconds = 1 / camera.rate_hz
        self.lags = np.array(camera.history_lags_frames)
        self.random = random  # the noise source; the env swaps it when it is reseeded

        oldest_needed_s = camera.latency_s + (self.lags.max() + 2) * self.frame_seconds
        self._ticks_kept = int(np.ceil(oldest_needed_s / physics_dt)) + 1
        self._positions = np.zeros((self._ticks_kept, objects, 2))  # ring buffer: slot = tick number % size
        self._tick = 0  # physics ticks since reset; tick 0 is the reset position

    # ── Record ──

    def reset(self, positions) -> None:
        """Start a new episode: before time 0 everything was sitting at `positions` (objects, 2)."""
        self._positions[:] = positions
        self._tick = 0

    def record(self, trace: np.ndarray) -> None:
        """Add the positions of one control tick, one (objects, 2) row per physics tick."""
        for positions in trace:
            self._tick += 1
            self._positions[self._tick % self._ticks_kept] = positions

    # ── Look ──

    def latest_frames(self) -> np.ndarray:
        """Noisy positions in the newest frame and the older ones the policy asked for."""
        newest_frame = np.floor((self._tick * self.physics_dt - self.camera.latency_s) / self.frame_seconds)
        frame_times = (newest_frame - self.lags) * self.frame_seconds
        ticks = np.clip(np.round(frame_times / self.physics_dt), 0, self._tick).astype(int)
        positions = self._positions[ticks % self._ticks_kept]
        return positions + self.random.normal(0.0, self.camera.puck_noise_m, positions.shape)
