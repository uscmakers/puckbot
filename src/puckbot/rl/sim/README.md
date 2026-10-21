# Sim

A simple MuJoCo model of our table, puck and paddle, plus a few hand-written skills.
No learning yet: this is the baseline a trained policy has to beat, and the world it will train in.

## Demo

Rendered with MuJoCo's own renderer. The robot (light disc) defends the left goal. These videos predate the
simple-mallet paddle model below; re-render them with the sbatch script when needed.

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
Loads the CAD model and clones its puck and paddle as many times as the `Layout` asks
(`Layout.of(paddles, pucks)`; default one robot paddle and one puck; `Layout` also sets each paddle's
side and reach: its own half, the full table, or a box). The pucks and paddles slide flat on the surface
(no tipping or lifting) and ignore the deck, as if on the air cushion. They still hit the rails and each
other. The table size is read back from the CAD, so it is never typed in twice. The physics does not
depend on the layout: add a paddle or puck and nothing else changes.

### 2. World: how does it move?
`env.step(targets)` gives each paddle an (x, y) target and runs 20 ms of physics. MuJoCo does all the
physics. A paddle is just a mass on a critically damped spring (a MuJoCo position actuator) that pulls it
after a target point. The point moves toward the target in a straight line within a top speed and a
moderate acceleration limit, braking so it stops on the target (`paddle_motion.py`; MuJoCo has no
equivalent for that planning). No gantry or stepper model yet: it behaves like a regular mallet, and
motor details come later once we measure them. The pucks' slow-down is
joint damping. A goal is a puck past the slot exit. Paddle 0 is the robot's: it defends -x and attacks +x;
an opponent paddle is the mirror image, with the same limits. Each paddle reaches 3 ft from its own end
rail (`reach_from_end_m`), the full width minus the wall margin. States are arrays: `state.puck_pos` is
(pucks, 2), `state.paddle_pos` is (paddles, 2), and `state.touches[paddle, puck]` says who hit what.
Units are metres, in the CAD frame (origin at the centre).

### 3. Skills: what should the paddle do?
Plain rules that look at puck 0 and one paddle and return a target. Nothing is learned. A skill takes
`paddle=` (which paddle it drives); for the opponent's paddle it mirrors x so the same rules work at either end.
It ignores other paddles, so it works with one or two. Every skill has the same interface and is batched
(torch), so this one implementation drives the CPU sim (`skill.target_one(state)`) and mjlab's thousands
of worlds (`skill.target(snapshot)`); `skill.reset(env_ids)` starts new episodes.
- `hold`: wait at home
- `straight_shot`, `bank_shot`: wind up behind the puck, then hit it toward an aim point
- `goalie`: stand where the puck will cross our defence line, side-rail bounces included

### 4. Drills: try a skill
A drill is a skill plus a starting puck. Running it records every tick, so it can be saved and replayed.

### 5. Baselines: how good is a shot from everywhere?
`puckbot-sim --grid` shoots a resting puck from a 4 x 7 grid of start spots on our half and prints a map.
The shots ignore any goalie, so with an empty goal this measures aim only; with `--goalie` the scripted goalie
(reads the true puck state, no delay) defends the far goal.

## Files

| File | Job |
| --- | --- |
| `settings.py` + `configs/sim.json` | The numbers we can tune (paddle speed, accel and response, rail bounce, timestep). Typos are rejected. |
| `paddle_motion.py` | How the target point moves: speed and acceleration limits, planned braking. |
| `layout.py` | How many paddles and pucks, and each paddle's side and reach. |
| `model.py` | Builds the MuJoCo model from the CAD for a layout. |
| `env.py` | `reset()`, `step(targets)`, goal and touch detection. |
| `skills.py` | The scripted skills: one batched interface for the CPU sim and mjlab. |
| `drills.py` | Runs one skill from a start and records it. |
| `baselines.py` | Score map of a shot skill over a grid of start spots. |
| `video.py` | Replays a run into an H.264 mp4 with MuJoCo's renderer. |
| `showcase.py` | Demo: two skills play a short match as points (serve, shot, block), saved as one mp4. |
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
puckbot-sim --grid                       # shot score map from many start spots (empty goal)
puckbot-sim --grid --goalie              # the same with a scripted goalie defending the far goal
```

Video needs OpenGL, which CARC login and CPU nodes don't have. MuJoCo's renderer works on a GPU node
through EGL (`MUJOCO_GL=egl`). Change the `--account` line in the script to yours, then:

```bash
mkdir -p outputs && sbatch src/puckbot/rl/sim/slurm/render_drills.sbatch   # writes outputs/<drill>.mp4
```

To refresh the demo videos, copy the new ones from `outputs/` into `demo/`.

## Baseline results

Each hand-written shot from a resting puck on a 4 x 7 grid of start spots on our half (`puckbot-sim --grid`,
add `--goalie` for the scripted goalie at the far goal). Physics values are still placeholders (see below).

| Skill | Empty goal | Mean time, puck speed at goal | vs scripted goalie |
| --- | --- | --- | --- |
| `straight_shot` | 28 / 28 scored | 0.82 s, 3.2 m/s | 0 / 28 |
| `bank_shot` | 28 / 28 scored | 0.84 s, 3.1 m/s | 0 / 28 |

- The shots keep their wind-up and follow-through inside the paddle's reach, so even pucks at the
  side rails or near the 3 ft reach limit go in.
- The goalie blocks every shot: it reads the true puck state with no delay, and the shots are slow.
- The env runs about 1560 control steps/s on one CPU core (31 simulated seconds per real second).

## What is approximate

These are placeholders until we have recordings from the real table.

- **Puck slow-down** is MuJoCo's joint damping (`slowdown_ns_per_m`), not a fitted friction + drag.
- **Rail bounce** is a soft contact tuned by eye. A square-on hit keeps 0.64-0.70 of its speed, a 30 deg
  hit keeps 0.78 (prior teams measured 0.785). `bounce_damping_ratio` below 0.1, or a timestep above
  0.5 ms, makes the contact unstable: the puck gains energy.
- **Bank shot** assumes the 0.78 (`RAIL_BOUNCE_KEEP` in `skills.py`). It scores from all 28 grid
  start spots with an empty goal, but it is not tuned beyond that.
- **Speed:** about 1650 control steps/s on one core (33 simulated seconds per real second) with one paddle.
- **Paddle:** the top speed (3 m/s), acceleration limit (40 m/s^2, the middle of the 20-60 m/s^2 range
  misprit7 tested; Hudson's reports give no number), response time (5 ms: it trails its target by about
  3 cm at top speed) and the 5 cm wall margin are guesses. No camera or command delay and no noise yet.
- **Known issue:** in random 1-10 m/s shots the puck sometimes gains speed near the goal mouths
  (the rail corners beside the slots). The drills stay under 3 m/s and don't trigger it. Look into it
  before training.
