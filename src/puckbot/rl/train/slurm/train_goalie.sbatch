#!/bin/bash
# Train the goalie with PPO on CPU cores (one process per env). No GPU needed.
# Submit from the repo root:  mkdir -p outputs && sbatch src/puckbot/rl/train/slurm/train_goalie.sbatch
# Results: outputs/train/goalie_ppo.zip + checkpoints (job log: outputs/train-<jobid>.out)
# Change --account to yours: sacctmgr show assoc user=$USER format=account
#SBATCH --job-name=puckbot-train
#SBATCH --account=biyik_1165
#SBATCH --partition=epyc-64
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/train-%j.out

module purge
export OMP_NUM_THREADS=1  # one thread per env process; the envs are the parallelism
cd "$SLURM_SUBMIT_DIR"
.venv/bin/python -P -m puckbot.rl.train.cli train --envs "$SLURM_CPUS_PER_TASK"
