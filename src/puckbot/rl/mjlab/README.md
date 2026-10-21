# mjlab

The game on GPU: mjlab (MuJoCo Warp physics + RSL-RL PPO), thousands of worlds at once. Our policy plays
paddle 0; the opponent paddle plays one of the scripted skills, drawn per world each episode. It uses the
same model, paddle rules and scripted skills as the CPU sim ([`../sim/`](../sim/README.md)): the skills are
one batched implementation (`sim/skills.py`) shared by both.

## Pipeline

```
sim/model.py build_spec ──► scene.py ──► mjlab env (game.py) ──► RSL-RL PPO ──► checkpoints
  (CAD table, pucks,        one entity,     actions.py   ours: (dx, dy) -> planner.py -> paddle spring
   paddles, springs)        touch sensors,               opponent: a scripted skill -> planner -> spring
                            camera          camera.py    120 fps frames, latency, lags (a sensor)
                                            events.py    pick the opponent's skill, place the puck to match
                                            game.py      +1 scored, -1 conceded, approach shaping; ends on a goal / 5 s
```

### Timing
Physics 2000 Hz (0.5 ms: rail bounces gain energy at coarser steps), camera 120 fps, policy 50 Hz
(40 physics ticks per action). The camera sensor records the puck every physics tick and hands the
policy the newest arrived frame plus older ones (lags 0, 1, 2, 5, 11 frames), as on the real robot.

### Episodes
Opponents come from `game.opponents` in `train/configs/train.json` (goalie, straight_shot, bank_shot). A
shooter gets the puck at rest on its half, so our policy defends; the goalie (or no opponent, with
`game.paddles = 1`) leaves the puck on our half, so our policy attacks. Skills only look at their own
paddle and puck 0, so the same code works with one paddle or two.

### What the policy sees and does
Same as the CPU env: camera frames of the pucks and opponent paddles, our paddle x, y, vx, vy, and the
previous action; (dx, dy) nudges the paddle's goal, 1 = one tick at top speed. Top speed 3 m/s,
acceleration 40 m/s^2, reach 3 ft from the end rail. PPO uses mjlab / RSL-RL defaults.

## Files

| File | Job |
| --- | --- |
| `scene.py` | Our model as one mjlab entity, touch sensors, the camera; `TableFacts` (sizes, limits, homes). |
| `planner.py` | The paddle's speed/acceleration-limited goal point, batched. |
| `actions.py` | Our paddle (dx, dy); the opponent paddle playing its scripted skill. |
| `camera.py` | 120 fps camera with latency and history lags, as an mjlab sensor. |
| `observations.py` | Camera frames (+ noise), our paddle state. |
| `state.py` | Puck and paddle positions as tensors; goal check. |
| `events.py` | Each episode: pick the opponent's skill, place the puck to match. |
| `game.py` | The game task: rewards, episode ends, env and PPO config. |
| `tasks.py`, `train.py`, `play.py` | Register `Puckbot-Game` and run mjlab's own train / play. |
| `record.py` | A checkpoint playing one world, saved as an mp4 (no viewer window; CPU or GPU). |
| `check_game.py` | Sanity check per opponent: doing nothing vs our paddle playing the matching skill. |
| `warp_skill_videos.py` | The CPU sim's skills in MuJoCo Warp, compared with CPU MuJoCo, rendered. |
| `slurm/` | `train_game.sbatch` (GPU, records a clip every 50 iterations), `warp_skill_videos.sbatch`. |
| `tests/` | The batched pieces match the CPU code (planner, camera timing). |

## Run

mjlab lives in its own venv (6 GB of CUDA libraries, so on scratch2):

```bash
export UV_CACHE_DIR=/scratch2/$USER/uv-cache UV_LINK_MODE=copy
uv venv /scratch2/$USER/venvs/mjlab --python 3.11
uv pip install --python /scratch2/$USER/venvs/mjlab/bin/python mjlab -e ".[sim]" pytest
PY=/scratch2/$USER/venvs/mjlab/bin/python

$PY -m pytest src/puckbot/rl/mjlab/tests                 # batched pieces match the CPU code
$PY -m puckbot.rl.mjlab.check_game cpu                    # works on any CPU, slowly
mkdir -p outputs && sbatch src/puckbot/rl/mjlab/slurm/train_game.sbatch 1500   # train on a GPU
$PY -m puckbot.rl.mjlab.play Puckbot-Game --checkpoint-file outputs/rsl_rl/puckbot_game/<run>/model_1500.pt
```

`python -m puckbot.rl.mjlab.record <run>/model_199.pt` saves a one-world mp4 (OpenGL setup in its docstring).
`play` also runs on a Mac (CPU, slow). ONNX export for the robot is not wired up yet (mjlab's runner
has `export_policy_to_onnx`; nothing calls it for our task).

## What is approximate

- Warp makes one contact point for puck-rail and puck-paddle hits (regular MuJoCo can make several), so
  paths after bounces drift by centimetres from the CPU sim. Outcomes matched in all 12 test shots.
- Physics values are still placeholders until the table is measured (see the sim README).
- The puck still sometimes gains speed at the goal-mouth corners above ~4 m/s (not fixed yet).
- Rewards are deliberately simple (score / concede / approach) until we settle the reward design. The first
  200-iteration run (2026-10-18) learned to stop conceding but parks at its reach limit: the approach
  shaping is gamed, so the reward design is the next fix.
