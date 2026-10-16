#!/bin/bash
# Run the drills that move something (not `hold`) and render an mp4 of each with MuJoCo's own renderer (needs a GPU for EGL).
# Submit from the repo root:  mkdir -p outputs && sbatch src/puckbot/sim/slurm/render_drills.sbatch
# Results: outputs/<drill>.npz and outputs/<drill>.mp4 (job log: outputs/render-<jobid>.out)
# Change --account to yours: sacctmgr show assoc user=$USER format=account
#SBATCH --job-name=puckbot-render
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=outputs/render-%j.out

module purge
export MUJOCO_GL=egl
cd "$SLURM_SUBMIT_DIR"
.venv/bin/python -P -m puckbot.sim.cli straight_shot bank_shot goalie --out-dir outputs --video
