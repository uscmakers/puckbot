# Train

PPO learns to play in the MuJoCo sim ([`../sim/`](../sim/README.md)).
First task: goalie, stopping random shots at our goal. There is ONE game env for every task: the number of
paddles and pucks and the scripted opponents come from the settings, and a task decides the starts and the reward.

## Pipeline

```
sim/ AirHockeyEnv ──► camera.py ──► observation.py ──► policy (PPO) ──► action.py ──► sim/
 (pucks + paddles)    120 fps,      object history +   "dx, dy"        nudge the
                      latency,      our paddle + last                  commanded point
                      noise         action
        game_env.py = all of the above in one Gymnasium env, plus scripted opponents for other paddles
        tasks.py: where episodes start, reward, when they end (goalie uses serves.py)
```

### 1. What the policy sees (`camera.py`, `observation.py`)
- **Camera:** 120 fps, 15 ms latency, 1.5 mm noise. The policy runs at 50 Hz, so each tick it takes the
  newest frame that has arrived (0-8 ms old, varying tick to tick, like on the real robot).
- **History:** every tracked object (the pucks, then the other paddles) in the newest frame and 1, 2, 5 and
  11 frames older (Hudson's choice): 0-92 ms. Raw positions, not velocities: a velocity estimate is wrong
  right after a bounce.
- **Also:** our paddle x, y, vx, vy and the previous action. 16 numbers with one puck and no opponent
  (26 with an opponent paddle), scaled to about [-1, 1].

### 2. What the policy does (`action.py`)
Two numbers (dx, dy) in [-1, 1]. They nudge the paddle's commanded point: 1 = one tick at top speed
(6 cm), 0 = hold. The sim's top speed (3 m/s) still applies.

### 3. The env and the task (`game_env.py`, `tasks.py`, `serves.py`)
`GameEnv` is the one env. Settings `game` say how many paddles and pucks (`Layout.of(paddles, pucks)`) and
which scripted skills the other paddles play (one drawn per episode). A task (`Task`: `reset` + `judge`) says
where things start and what is rewarded. The first task, `GoalieTask`:
- **Start:** a random serve: far half, random speed (1-4 m/s), aimed at our goal or via a side rail (30%).
- **Reward:** +1 blocked, -1 conceded, plus a small shaping term for closing the gap between the paddle
  and the puck sideways (potential-based, so it cannot be farmed).
- **Ends:** blocked, conceded, missed (puck bounced off the end rail untouched), or 5 s timeout.

A new task (a shooter, the full game) is a new class in `tasks.py`, not a new env.

### 4. Learning (`train.py`)
Stable-Baselines3 PPO with **all default hyperparameters** and its default 2x64 network. We only choose
the number of parallel envs, total steps and seed. Change a default only for a clear reason.

### 5. Judging (`evaluate.py`)
Training reward is not the score. `evaluate` plays the same 200 fixed serves for any policy (a model,
the hand-written goalie, or `hold`) and counts blocked / conceded / missed / timeout.

## Files

| File | Job |
| --- | --- |
| `settings.py` + `configs/train.json` | Camera, game layout, task and run settings. Typos are rejected. |
| `camera.py` | Delay, noise and frame timing of the tracked objects' positions. |
| `observation.py` | Turns camera frames + paddle state into the policy's input vector. |
| `action.py` | Turns the policy's (dx, dy) into a paddle target (and back). |
| `serves.py` | Random shots at our goal. |
| `game_env.py` | The one Gymnasium env: layout from settings, scripted opponents, camera, observation. |
| `tasks.py` | What to do: `GoalieTask` (starts, reward, episode end). New tasks go here. |
| `train.py` | PPO training loop and checkpoints. |
| `evaluate.py` | Fixed-serve scoring for models and hand-written skills. |
| `cli.py` | The `puckbot-train` command. |
| `slurm/train_goalie.sbatch` | Trains on 32 CPU cores of a CARC node. |
| `tests/` | Camera timing, action, serves, the env with any layout or task, a tiny PPO run. |

## Run

```bash
# once: CPU torch first (plain `pip install torch` pulls a multi-GB CUDA build)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[train]"

source .venv/bin/activate
puckbot-train evaluate --skill hold          # the floor: doing nothing
puckbot-train evaluate --skill goalie        # the hand-written baseline to beat
mkdir -p outputs && sbatch src/puckbot/rl/train/slurm/train_goalie.sbatch   # or: puckbot-train train
puckbot-train evaluate --model outputs/train/goalie_ppo.zip
```

On CARC, train on a compute node, not a login node. Speed on one core, camera and observation included:
about 1100 env steps/s for the goalie task (22 simulated seconds per real second), 750 with an opponent
paddle. With 32 cores, if it scales linearly (not measured), that is tens of thousands of steps/s.

## What is approximate

- **Camera latency (15 ms) and noise** are guesses (Hudson measured 13 ms and 1.5 mm). Measure ours
  with the LED test, then change `camera` in `train.json`.
- **Serve speeds** stop at 4 m/s: above that the sim's puck sometimes gains speed near the goal
  mouths (see the sim README). Fix that before raising them.
- **No dropouts, no physics randomisation** yet, and scripted opponents read the true state (no delay). A policy trained like this may be brittle.
- **Left out on purpose:** reflecting the table (mirror augmentation), ONNX export, shot-type request.
  They come with the shooter and the full game.
