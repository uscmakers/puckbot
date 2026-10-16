# RL

Everything for learning to play: the simulated table and the training code.

| Folder | What it is |
| --- | --- |
| [`sim/`](sim/README.md) | MuJoCo model of our table, puck and paddle, plus hand-written skills (the baseline). |
| [`train/`](train/README.md) | The task as a Gymnasium env, what the policy sees and does, PPO training, evaluation. |

`train/` uses `sim/`; `sim/` knows nothing about learning.
The policy network and its training stay together in `train/`: the network is one line (SB3's default
`MlpPolicy`), so a separate model folder would only hold that line. Split it out when the network or
the ONNX export grows.
