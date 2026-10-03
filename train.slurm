#!/bin/bash
# Template: edit partition, account and GPU resource options for your cluster.
#SBATCH --job-name=dm-day2
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Add your cluster's module/container setup here if required.
# Set DM_PYTHON before submission to use a different environment.
DM_PYTHON="${DM_PYTHON:-${SLURM_SUBMIT_DIR}/.venv/bin/python}"
srun "${DM_PYTHON}" -u train.py --device cuda --steps 5000 --batch-size 256 --output outputs/run1
srun "${DM_PYTHON}" -u sample.py --device cuda --checkpoint outputs/run1/model.pt
