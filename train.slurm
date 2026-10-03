#!/bin/bash
#SBATCH --job-name=dm-test-model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liubixuan214@gmail.com

# SCAI / ISIR: https://scai.sorbonne-universite.fr/computing-power
# The public page specifies A100 hardware, but not Slurm resource names.
# "a100" above is an assumed GRES type; confirm it on the cluster with:
#   sinfo -N -o '%P %N %G %f'
# No partition/account/QoS is hard-coded: use your site's defaults or pass
# --partition, --account and --qos to sbatch as required for your access.
# If the A100 GRES type differs, override --gres=gpu:EXACT_TYPE:1 at submission.
# For untyped GPUs, select a confirmed A100-only partition/node/constraint
# and override --gres=gpu:1. The runtime check below verifies the GPU model.
# Submit from the DM_ex directory, after creating the Slurm log directory:
#   mkdir -p outputs
#   sbatch --mail-user=YOUR_EMAIL train.slurm
# Slurm opens its log before this script runs, so mkdir must precede sbatch.
# END/FAIL request email when the whole job ends or fails, including sampling.
# Fill and uncomment --mail-user above, or set it at submission. Without it, Slurm uses
# the submitting username and the cluster's configured mail domain.
# Notifications require the cluster's Slurm mail service to be configured.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Use your CUDA-enabled PyTorch environment. If SCAI requires modules or a
# container, apply its documented setup here; no module names are assumed.
# Override the interpreter with:
#   DM_PYTHON=/absolute/path/to/env/bin/python sbatch train.slurm
DM_PYTHON="${DM_PYTHON:-${SLURM_SUBMIT_DIR}/.venv/bin/python}"
if [[ ! -x "${DM_PYTHON}" ]]; then
    echo "Python is not executable: ${DM_PYTHON}. Set DM_PYTHON to your environment's interpreter." >&2
    exit 1
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
# Keep Slurm's CUDA_VISIBLE_DEVICES unchanged; it identifies the allocated GPU.

# Check inside a job step, where Slurm applies GPU visibility and isolation.
srun --ntasks=1 "${DM_PYTHON}" -u - <<'PY'
import torch

if int(torch.__version__.split('.')[0]) < 2:
    raise SystemExit('This project requires PyTorch >= 2.0; select a compatible environment.')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in this job; check your PyTorch environment and GPU allocation.')
if torch.cuda.device_count() != 1:
    raise SystemExit(f'Expected exactly one visible GPU, got {torch.cuda.device_count()}.')
name = torch.cuda.get_device_name(0)
if 'A100' not in name.upper() or 'MIG' in name.upper():
    raise SystemExit(f'Expected a full NVIDIA A100 GPU, got {name}. Check your Slurm resource selection.')
print(f'PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; GPU: {name}')
PY

srun --ntasks=1 "${DM_PYTHON}" -u train.py --device cuda --steps 5000 --batch-size 256 --output outputs/run1
srun --ntasks=1 "${DM_PYTHON}" -u sample.py --device cuda --checkpoint outputs/run1/model.pt
