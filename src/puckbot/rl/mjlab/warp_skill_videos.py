"""Run the hand-written skills in MuJoCo Warp (GPU), check them against regular MuJoCo, and make videos.

Each shot is one Warp "world": all of them step together on the GPU. The skills and the paddle
target planner are the same Python code as the CPU sim; only the physics runs in Warp.

Steps:
    1. Build     our model (sim/model.py) and copy it to Warp with one world per shot
    2. Run       every control tick: read the state, each skill picks a target, the planner moves
                 the target point over the next 40 physics ticks, Warp steps
    3. Check     run the same shots in regular MuJoCo (CPU) and compare outcome and puck path
    4. Render    one mp4 per shot plus one grid video of all of them (needs MUJOCO_GL=egl)

Usage (mjlab venv):
    python -m puckbot.rl.mjlab.warp_skill_videos --out-dir outputs/warp_videos           # GPU
    python -m puckbot.rl.mjlab.warp_skill_videos --device cpu --seconds 0.5 --no-video   # quick check
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from puckbot.rl.sim.drills import run_skill
from puckbot.rl.sim.env import AirHockeyEnv, State
from puckbot.rl.sim.layout import Layout
from puckbot.rl.sim.paddle_motion import PaddleMotion
from puckbot.rl.sim.skills import make_skill

# ── Shots ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Shot:
    name: str
    skill: str  # robot's skill (paddle 0)
    puck_pos: tuple[float, float]
    puck_vel: tuple[float, float] = (0.0, 0.0)
    opponent: str | None = None  # skill for paddle 1 (two-paddle table)


SHOTS = (
    Shot("straight_centre", "straight_shot", (-0.40, 0.00)),
    Shot("straight_left", "straight_shot", (-0.40, 0.27)),
    Shot("straight_right", "straight_shot", (-0.55, -0.27)),
    Shot("straight_close", "straight_shot", (-0.25, 0.13)),
    Shot("bank_left", "bank_shot", (-0.40, 0.15)),
    Shot("bank_right", "bank_shot", (-0.55, -0.27)),
    Shot("bank_far", "bank_shot", (-0.70, 0.27)),
    Shot("goalie_straight", "goalie", (0.60, 0.00), (-2.5, 0.0)),
    Shot("goalie_angled", "goalie", (0.60, 0.25), (-2.0, -0.35)),
    Shot("goalie_off_the_rail", "goalie", (0.50, 0.30), (-2.0, 1.2)),
    Shot("straight_vs_goalie", "straight_shot", (-0.40, 0.15), opponent="goalie"),
    Shot("bank_vs_goalie", "bank_shot", (-0.40, 0.15), opponent="goalie"),
)


# ── Run in Warp ───────────────────────────────────────────────────────────────


@dataclass
class WarpRun:
    shot: Shot
    time: np.ndarray  # (n,)
    qpos: np.ndarray  # (n, nq)
    outcome: str


def run_in_warp(
    shots: list[Shot], layout: Layout, seconds: float, device: str
) -> tuple[list[WarpRun], float]:
    """All shots on one table layout, one Warp world each. Returns the runs and physics ticks per second."""
    env = AirHockeyEnv(layout=layout)  # the CPU env supplies the model, geometry, homes and limits
    model, worlds = env.model, len(shots)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    skills = [
        [make_skill(s.skill, env)] + ([make_skill(s.opponent, env, paddle=1)] if s.opponent else [])
        for s in shots
    ]
    skills = [
        row + [make_skill("hold", env, paddle=k) for k in range(len(row), env.n_paddles)] for row in skills
    ]
    motions = [
        [
            PaddleMotion(env.settings.paddle, env.paddle_low[k], env.paddle_high[k], env.home[k])
            for k in range(env.n_paddles)
        ]
        for _ in shots
    ]
    for row in skills:
        for skill in row:
            skill.reset()

    qpos0 = np.tile(data.qpos, (worlds, 1))
    qvel0 = np.zeros((worlds, model.nv))
    for w, shot in enumerate(shots):
        qpos0[w, env._puck_qpos[0]] = shot.puck_pos
        qvel0[w, env._puck_dof[0]] = shot.puck_vel
        qpos0[w, env._paddle_qpos.ravel()] = env.home.ravel()
    ctrl0 = np.tile(env.home.ravel(), (worlds, 1))

    ticks = env.physics_ticks_per_step
    with wp.ScopedDevice(device):
        warp_model = mjw.put_model(model)
        warp_data = mjw.put_data(model, data, nworld=worlds, nconmax=64, njmax=256, naconmax=64 * worlds)
        warp_data.qpos.assign(qpos0)
        warp_data.qvel.assign(qvel0)
        warp_data.ctrl.assign(ctrl0)
        mjw.forward(warp_model, warp_data)

        frames, times, outcomes = [qpos0.copy()], [0.0], [None] * worlds
        stepping_seconds = 0.0
        for step in range(round(seconds / env.step_seconds)):
            qpos, qvel = warp_data.qpos.numpy(), warp_data.qvel.numpy()
            ctrl = np.empty((ticks, worlds, model.nu))
            for w in range(worlds):
                state = _state(env, qpos[w], qvel[w])
                outcomes[w] = outcomes[w] or state.goal
                for k, (skill, motion) in enumerate(zip(skills[w], motions[w], strict=True)):
                    target = skill.target_one(state)
                    for tick in range(ticks):
                        motion.advance(target[0], target[1], env.physics_dt)
                        ctrl[tick, w, 2 * k : 2 * k + 2] = motion.x, motion.y
            started = time.perf_counter()
            for tick in range(ticks):
                warp_data.ctrl.assign(ctrl[tick])
                mjw.step(warp_model, warp_data)
            wp.synchronize()
            stepping_seconds += time.perf_counter() - started
            frames.append(warp_data.qpos.numpy().copy())
            times.append((step + 1) * env.step_seconds)

    qpos_by_world = np.stack(frames, axis=1)  # (worlds, n, nq)
    runs = [
        WarpRun(shot, np.array(times), qpos_by_world[w], outcomes[w] or "no goal")
        for w, shot in enumerate(shots)
    ]
    physics_ticks = worlds * ticks * (len(times) - 1)
    return runs, physics_ticks / max(stepping_seconds, 1e-9)


def _state(env: AirHockeyEnv, qpos: np.ndarray, qvel: np.ndarray) -> State:
    puck_pos = qpos[env._puck_qpos]
    return State(
        time=0.0,
        puck_pos=puck_pos,
        puck_vel=qvel[env._puck_dof],
        paddle_pos=qpos[env._paddle_qpos],
        paddle_vel=qvel[env._paddle_dof],
        goals=tuple(env._goal_at(*position) for position in puck_pos),
        touches=np.zeros((env.n_paddles, env.n_pucks), dtype=bool),
    )


# ── Check against regular MuJoCo ──────────────────────────────────────────────


def compare_with_cpu(runs: list[WarpRun], layout: Layout, seconds: float) -> None:
    env = AirHockeyEnv(layout=layout)
    puck = env._puck_qpos[0]
    for run in runs:
        shot = run.shot
        cpu = run_skill(env, shot.skill, shot.puck_pos, shot.puck_vel, seconds, opponent=shot.opponent)
        n = min(len(cpu.qpos), len(run.qpos))
        until_goal = np.abs(cpu.qpos[:n, puck] - run.qpos[:n, puck]).max() * 1000
        print(
            f"  {shot.name:22s} Warp {run.outcome:9s} CPU {cpu.outcome:9s} "
            f"puck paths differ by <= {until_goal:6.1f} mm"
        )


# ── Videos ────────────────────────────────────────────────────────────────────


def write_videos(runs: list[WarpRun], model: mujoco.MjModel, out_dir: Path) -> None:
    from puckbot.rl.sim.video import write_video  # needs OpenGL (MUJOCO_GL=egl)

    for run in runs:
        write_video(model, SimpleNamespace(time=run.time, qpos=run.qpos), out_dir / f"{run.shot.name}.mp4")


def write_grid_video(runs: list[WarpRun], models: dict, out_dir: Path, columns: int = 4) -> None:
    """All shots side by side, each labelled with its name and outcome."""
    import cv2
    import imageio_ffmpeg

    from puckbot.rl.sim.video import _top_down_camera

    width, height = 480, 270
    rows = -(-len(runs) // columns)
    renderers = {id(m): mujoco.Renderer(m, height=height, width=width) for m in models.values()}
    datas = {id(m): mujoco.MjData(m) for m in models.values()}
    camera = _top_down_camera()
    length = max(len(run.qpos) for run in runs)
    fps = round(1 / float(runs[0].time[1] - runs[0].time[0]))
    writer = imageio_ffmpeg.write_frames(
        str(out_dir / "all_shots.mp4"),
        (columns * width, rows * height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
    )
    writer.send(None)
    try:
        for frame in range(length):
            canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
            for index, run in enumerate(runs):
                model = models[run.shot.opponent is not None]
                data = datas[id(model)]
                data.qpos[:] = run.qpos[min(frame, len(run.qpos) - 1)]
                mujoco.mj_forward(model, data)
                renderer = renderers[id(model)]
                renderer.update_scene(data, camera)
                image = np.ascontiguousarray(renderer.render())
                cv2.putText(
                    image,
                    f"{run.shot.name}  [{run.outcome}]",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                row, column = divmod(index, columns)
                canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
            writer.send(canvas.tobytes())
    finally:
        writer.close()
        for renderer in renderers.values():
            renderer.close()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda:0", help="Warp device: cuda:0 or cpu")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--out-dir", default="outputs/warp_videos")
    parser.add_argument("--no-video", action="store_true", help="skip rendering (no OpenGL needed)")
    args = parser.parse_args()

    wp.init()
    print("Warp device:", args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs, models = [], {}
    for two_paddles in (False, True):
        shots = [s for s in SHOTS if (s.opponent is not None) == two_paddles]
        layout = Layout.of(2 if two_paddles else 1)
        runs, ticks_per_second = run_in_warp(shots, layout, args.seconds, args.device)
        print(
            f"{len(shots)} shots on a {len(layout.paddles)}-paddle table: "
            f"Warp ran {ticks_per_second:,.0f} physics ticks/s"
        )
        compare_with_cpu(runs, layout, args.seconds)
        all_runs += runs
        models[two_paddles] = AirHockeyEnv(layout=layout).model

    if not args.no_video:
        for two_paddles, model in models.items():
            write_videos(
                [r for r in all_runs if (r.shot.opponent is not None) == two_paddles], model, out_dir
            )
        write_grid_video(all_runs, models, out_dir)
        print("videos in", out_dir)


if __name__ == "__main__":
    main()
