"""The batched (GPU) pieces give the same answers as the CPU code they copy. Needs the mjlab venv."""

import numpy as np
import pytest

pytest.importorskip("mjlab")
import torch  # noqa: E402

from puckbot.rl.mjlab.camera import frame_ticks  # noqa: E402
from puckbot.rl.mjlab.planner import BatchedPlanner  # noqa: E402
from puckbot.rl.mjlab.scene import TableFacts  # noqa: E402
from puckbot.rl.sim.layout import Layout  # noqa: E402
from puckbot.rl.sim.paddle_motion import PaddleMotion  # noqa: E402
from puckbot.rl.train.camera import CameraModel  # noqa: E402
from puckbot.rl.train.settings import CameraSettings  # noqa: E402


@pytest.fixture(scope="module")
def facts():
    return TableFacts.load(Layout.of(2))


def test_batched_planner_matches_the_cpu_planner(facts):
    paddle = facts.settings.paddle
    rng = np.random.default_rng(0)
    goals = rng.uniform(facts.paddle_low[0], facts.paddle_high[0], size=(3, 2))
    cpu = [PaddleMotion(paddle, facts.paddle_low[0], facts.paddle_high[0], facts.home[0]) for _ in goals]
    batched = BatchedPlanner(
        3, facts.paddle_low[0], facts.paddle_high[0], paddle.max_speed_mps, paddle.max_accel_mps2, "cpu"
    )
    batched.reset(slice(None), torch.as_tensor(facts.home[0], dtype=torch.float32))
    for _ in range(800):
        for motion, goal in zip(cpu, goals, strict=True):
            motion.advance(*goal, facts.physics_dt)
        batched.advance(torch.as_tensor(goals, dtype=torch.float32), facts.physics_dt)
    expected = np.array([motion.position for motion in cpu])
    assert np.allclose(batched.position.numpy(), expected, atol=1e-4)


def test_camera_frame_timing_matches_the_cpu_camera():
    camera = CameraSettings(
        rate_hz=120, latency_s=0.015, puck_noise_m=0.0, history_lags_frames=(0, 1, 2, 5, 11)
    )
    dt = 0.0005
    cpu = CameraModel(camera, objects=1, physics_dt=dt, random=np.random.default_rng(0))
    cpu.reset(np.zeros((1, 2)))
    for tick in range(1, 400):
        cpu.record(np.array([[[tick * dt, 0.0]]]))  # x = time: the frame's x tells its tick
        if tick % 40 == 0:
            expected_ticks = np.round(cpu.latest_frames()[:, 0, 0] / dt).astype(int)
            got = frame_ticks(torch.tensor([tick]), dt, camera)[0].numpy()
            assert np.array_equal(got, expected_ticks)


def test_paddles_reset_to_home(facts):
    from mjlab.utils.string import resolve_expr

    from puckbot.rl.mjlab.scene import ENTITY, make_scene_cfg
    from puckbot.rl.sim.model import paddle_joints, puck_joints

    scene = make_scene_cfg(facts, num_envs=1, camera=CameraSettings(120, 0.0, 0.0, (0,)))
    names = (*paddle_joints(0), *paddle_joints(1), *puck_joints(0))
    start = resolve_expr(scene.entities[ENTITY].init_state.joint_pos, names, 0.0)
    assert np.allclose(start[:4], facts.home.reshape(-1))  # not the table centre (mjlab: first match wins)
    assert np.allclose(start[4:], 0.0)
