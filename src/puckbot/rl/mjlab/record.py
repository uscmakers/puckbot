"""Record a trained checkpoint playing the game to an mp4 (one world, top-down), with no viewer window.

Steps:
    1. Play      the Puckbot-Game play env with one world (episodes end only on goals) for --seconds,
                 saving every control tick's joint positions in the CPU model's layout (.npz)
    2. Render    in a fresh process, with the CPU sim's video writer (sim/video.py)
    3. Save      the mp4 next to the checkpoint (videos/record/) and print each goal

Two processes because software OpenGL (CPU nodes) and Triton (pulled in by rsl_rl) cannot share one.
`mjlab.play --video` also records, but then opens a viewer, which a batch job cannot use.

Usage (mjlab venv). OpenGL: on a GPU node export MUJOCO_GL=egl. On a CPU node (software OpenGL):
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/mesa-24.2.3-s7eilwc/lib:\
/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/gcc-runtime-13.3.0-cm7m2ub/lib
    python -m puckbot.rl.mjlab.record outputs/rsl_rl/puckbot_game/<run>/model_199.pt [--seconds 20]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

TASK = "Puckbot-Game"


# ── Play: run the policy, keep the joint positions ────────────────────────────


def play(checkpoint: Path, seconds: float, device: str) -> Path:
    from dataclasses import asdict

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import puckbot.rl.mjlab.tasks  # noqa: F401  (registers our tasks)
    from puckbot.rl.mjlab.scene import ENTITY
    from puckbot.rl.sim.model import build_model

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    agent_cfg = load_rl_cfg(TASK)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    facts = env.action_manager.get_term("paddle").cfg.facts
    cpu_model = build_model(facts.settings, facts.layout)
    entity = env.scene[ENTITY]
    slots = [cpu_model.joint(name).qposadr[0] for name in entity.joint_names]

    obs = vec_env.get_observations()
    qpos = []
    for step in range(int(seconds / env.step_dt)):
        with torch.no_grad():
            obs, _, dones, _ = vec_env.step(policy(obs))
        row = cpu_model.qpos0.copy()
        row[slots] = entity.data.joint_pos[0].cpu().numpy()
        qpos.append(row)
        if dones[0]:
            judge = env.unwrapped._game_judge
            print(f"{step * env.step_dt:5.1f} s  {'scored' if judge.scored[0] else 'conceded'}")

    path = checkpoint.parent / "videos" / "record" / f"{checkpoint.stem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, qpos=np.array(qpos), time=np.arange(len(qpos)) * env.step_dt)
    return path


# ── Render: a fresh process with OpenGL loaded first ──────────────────────────


def render(recording: Path) -> Path:
    if os.environ.get("MUJOCO_GL") == "osmesa":
        import OpenGL.GL  # noqa: F401  (software OpenGL before anything else loads its own libraries)

    from types import SimpleNamespace

    from puckbot.rl.mjlab.scene import TableFacts
    from puckbot.rl.sim.layout import Layout
    from puckbot.rl.sim.model import build_model
    from puckbot.rl.sim.video import write_video
    from puckbot.rl.train.settings import load_train_settings

    game = load_train_settings().game
    facts = TableFacts.load(Layout.of(game.paddles, game.pucks))
    saved = np.load(recording)
    out = recording.with_suffix(".mp4")
    write_video(build_model(facts.settings, facts.layout), SimpleNamespace(**saved), out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="model_N.pt, or a recording .npz with --render-only")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--device", default=None, help="default: cuda:0 if there is a GPU, else cpu")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    if args.render_only:
        print(f"saved {render(args.checkpoint)}")
        return
    render_env = dict(os.environ)
    for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):  # playing needs no OpenGL; keep it out of this process
        os.environ.pop(name, None)
    import torch

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    recording = play(args.checkpoint, args.seconds, device)
    command = [sys.executable, "-P", "-m", __spec__.name, str(recording), "--render-only"]
    subprocess.run(command, check=True, env=render_env)


if __name__ == "__main__":
    main()
