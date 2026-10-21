"""mjlab's `train` command, with our tasks registered.

Usage (mjlab venv):
    python -m puckbot.rl.mjlab.train Puckbot-Game --env.scene.num-envs 4096
"""

from mjlab.scripts.train import main

import puckbot.rl.mjlab.tasks  # noqa: F401  (registers our tasks)

if __name__ == "__main__":
    main()
