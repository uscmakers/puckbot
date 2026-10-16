"""Turn a recorded run into an mp4 using MuJoCo's own renderer.

Steps:
    1. Replay    set the model to each recorded tick (qpos)
    2. Render    mujoco.Renderer draws the CAD meshes from a top-down camera
    3. Write     frames go into an H.264 mp4 (plays in browsers and QuickTime)

Needs an OpenGL backend. CARC login/CPU nodes have none, so run on a GPU node with:
    export MUJOCO_GL=egl        (must be set before Python starts)

Usage:
    write_video(env.model, run, "run.mp4")
"""

from __future__ import annotations

import os
from pathlib import Path

import imageio_ffmpeg
import mujoco

from puckbot.rl.sim.drills import Run

TABLE_CENTRE = (0.0, 0.0, 0.74)  # middle of the playing surface (metres)


def write_video(
    model: mujoco.MjModel, run: Run, path: str | Path, width: int = 960, height: int = 540
) -> None:
    if not os.environ.get("MUJOCO_GL"):
        raise RuntimeError("No OpenGL backend set. On a GPU node run: export MUJOCO_GL=egl")

    fps = round(1 / float(run.time[1] - run.time[0]))
    data = mujoco.MjData(model)
    camera = _top_down_camera()
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
    )
    writer.send(None)  # start the encoder
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for qpos in run.qpos:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            writer.send(renderer.render().tobytes())
    finally:
        renderer.close()
        writer.close()


def _top_down_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = TABLE_CENTRE
    camera.distance = 1.9
    camera.azimuth = 90  # puts the robot's goal (-x) on the left
    camera.elevation = -90  # straight down
    return camera
