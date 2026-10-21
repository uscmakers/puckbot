#!/bin/bash
# Train the game policy in mjlab (MuJoCo Warp + RSL-RL PPO) on one GPU.
# Submit from the repo root:  mkdir -p outputs && sbatch src/puckbot/rl/mjlab/slurm/train_game.sbatch [ITERATIONS] [NUM_ENVS]
# Results: outputs/rsl_rl/puckbot_game/<run>/ (checkpoints, tensorboard, videos/train/); log: outputs/train-game-<jobid>.out
#SBATCH --job-name=puckbot-game
#SBATCH --account=biyik_1173
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/train-game-%j.out

module purge
export MUJOCO_GL=egl
export WARP_CACHE_PATH=/scratch2/ubhuwani/warp-cache
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv
/scratch2/ubhuwani/venvs/mjlab/bin/python -P -m puckbot.rl.mjlab.check_game cuda:0   # sanity check first
/scratch2/ubhuwani/venvs/mjlab/bin/python -P -m puckbot.rl.mjlab.train Puckbot-Game \
  --env.scene.num-envs "${2:-4096}" --agent.max-iterations "${1:-200}" --log-root outputs/rsl_rl \
  --video True --video-length 250 --video-interval 1200   # a 5 s clip every 50 iterations
