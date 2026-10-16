# Sim

A simple MuJoCo model of our table, puck and paddle, plus a few hand-written skills.
No learning yet: this is the baseline a trained policy has to beat, and the world it will train in.

## Demo

Rendered with MuJoCo's own renderer. The robot (light disc) defends the left goal.

| Skill | What it does | Video |
| --- | --- | --- |
| `straight_shot` | winds up behind a resting puck, hits it at the goal | [demo/straight_shot.mp4](demo/straight_shot.mp4) |
| `bank_shot` | same, aimed so the puck bounces off a side rail into the goal | [demo/bank_shot.mp4](demo/bank_shot.mp4) |
| `goalie` | blocks a shot coming at our goal | [demo/goalie.mp4](demo/goalie.mp4) |

## Pipeline

```
cad/table_urdf/table.xml ──► 1. Model ──► 2. World ◄── 3. Skill
  (CAD meshes + boxes)       model.py     env.py       skills.py
                             flat puck    reset()      "where should the
                             + paddle     step(x, y)    paddle go?"
                                             │
                    4. Drill: a skill + a starting setup (drills.py)
                                             │
                             Run (every tick) ──► .npz log ──► .mp4 (video.py)
```

### 1. Model: what is on the table?
Loads the CAD model. The puck and paddle slide flat on the surface (no tipping or lifting) and ignore
the deck, as if on the air cushion. They still hit the rails and each other. The table size is read
back from the CAD, so it is never typed in twice.

### 2. World: how does it move?
`env.step((x, y))` gives the paddle a target and runs 20 ms of physics. The paddle has a top speed and
an acceleration limit (`paddle_motion.py`), like the real gantry. A goal is a puck past the slot exit.
The robot defends -x and attacks +x. Units are metres, in the CAD frame (origin at the table centre).

### 3. Skills: what should the paddle do?
Plain rules that look at the puck and paddle and return a target. Nothing is learned.
- `hold`: wait at home
- `straight_shot`, `bank_shot`: wind up behind the puck, then hit it toward an aim point
- `goalie`: stand where the puck will cross our defence line, side-rail bounces included

### 4. Drills: try a skill
A drill is a skill plus a starting puck. Running it records every tick, so it can be saved and replayed.

## Files

| File | Job |
| --- | --- |
| `settings.py` + `configs/sim.json` | The numbers we can tune (paddle caps, rail bounce, timestep). Typos are rejected. |
| `model.py` | Builds the MuJoCo model from the CAD. |
| `paddle_motion.py` | Speed and acceleration limits. |
| `env.py` | `reset()`, `step(target)`, goal detection. |
| `skills.py` | The skills. |
| `drills.py` | Runs one skill from a fixed start and records it. |
| `video.py` | Replays a run into an H.264 mp4 with MuJoCo's renderer. |
| `cli.py` | The `puckbot-sim` command. |
| `slurm/render_drills.sbatch` | Renders the demo videos on a GPU node. |
| `demo/` | The videos above, for reference. |
| `tests/` | Physics checks and one test per skill. |

## Run

```bash
source .venv/bin/activate
puckbot-sim --list
puckbot-sim                              # every drill, prints the outcome (any node, no graphics)
puckbot-sim bank_shot --out-dir outputs  # + saves outputs/bank_shot.npz
```

Video needs OpenGL, which CARC login and CPU nodes don't have. MuJoCo's renderer works on a GPU node
through EGL (`MUJOCO_GL=egl`). Change the `--account` line in the script to yours, then:

```bash
mkdir -p outputs && sbatch src/puckbot/sim/slurm/render_drills.sbatch   # writes outputs/<drill>.mp4
```

To refresh the demo videos, copy the new ones from `outputs/` into `demo/`.

## What is approximate

These are placeholders until we have recordings from the real table.

- **Puck slow-down** is MuJoCo's joint damping (`slowdown_ns_per_m`), not a fitted friction + drag.
- **Rail bounce** is a soft contact tuned by eye. A square-on hit keeps 0.64-0.70 of its speed, a 30 deg
  hit keeps 0.78 (prior teams measured 0.785). `bounce_damping_ratio` below 0.1, or a timestep above
  0.5 ms, makes the contact unstable: the puck gains energy.
- **Bank shot** assumes the 0.78 (`RAIL_BOUNCE_KEEP` in `skills.py`). It scored from the 5 start
  positions we tried, but it is not tuned beyond that.
- **Paddle** is pulled after a commanded point by a spring-damper. The caps (3 m/s, 20 m/s^2) and the
  5 cm wall margin are guesses. No camera or command delay and no noise yet.
- **Known issue:** in random 1-10 m/s shots the puck sometimes gains speed near the goal mouths
  (the rail corners beside the slots). The drills stay under 3 m/s and don't trigger it. Look into it
  before training.
