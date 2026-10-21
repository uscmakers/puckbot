"""Register our mjlab tasks so mjlab's `train` / `play` can find them.

Usage:
    import puckbot.rl.mjlab.tasks       # then "Puckbot-Game" is in mjlab's registry
"""

from mjlab.tasks.registry import register_mjlab_task

from puckbot.rl.mjlab.game import game_env_cfg, game_ppo_cfg

register_mjlab_task(
    task_id="Puckbot-Game",
    env_cfg=game_env_cfg(),
    play_env_cfg=game_env_cfg(num_envs=16, play=True),
    rl_cfg=game_ppo_cfg(),
)
