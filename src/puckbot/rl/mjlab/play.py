"""mjlab's `play` command, with our tasks registered (works on a Mac for evaluation).

Usage (mjlab venv):
    python -m puckbot.rl.mjlab.play Puckbot-Game \
        --checkpoint-file outputs/rsl_rl/puckbot_game/<run>/model_1500.pt
"""

from mjlab.scripts.play import main

import puckbot.rl.mjlab.tasks  # noqa: F401  (registers our tasks)

if __name__ == "__main__":
    main()
