#!/bin/bash
# Hand-written skills in MuJoCo Warp on a GPU: compare with regular MuJoCo and render videos.
# Submit from the repo root:  mkdir -p outputs && sbatch src/puckbot/rl/mjlab/slurm/warp_skill_videos.sbatch
# Results: outputs/warp_videos/<shot>.mp4 and all_shots.mp4 (log: outputs/warp-<jobid>.out)
#SBATCH --job-name=puckbot-warp
#SBATCH --account=biyik_1173
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=outputs/warp-%j.out

module purge
export MUJOCO_GL=egl
export WARP_CACHE_PATH=/scratch2/ubhuwani/warp-cache
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
/scratch2/ubhuwani/venvs/mjlab/bin/python -P -m puckbot.rl.mjlab.warp_skill_videos --out-dir outputs/warp_videos
