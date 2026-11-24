#!/bin/bash
#SBATCH --job-name=pinn-tmp-smoke
#SBATCH --output=slurm-%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Node: $(hostname)"
echo "Compute-node temporary filesystem:"
findmnt -T /tmp
df -h /tmp

# The environment is created on this compute node, not under /home.
JOB_TMP=$(mktemp -d "/tmp/cylinder-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$JOB_TMP"' EXIT

BASE_PYTHON=/opt/ohpc/pub/apps/miniforge3/envs/jupyter/bin/python
"$BASE_PYTHON" -m venv "$JOB_TMP/env"
PYTHON="$JOB_TMP/env/bin/python"

"$PYTHON" -m pip install --no-cache-dir numpy matplotlib
"$PYTHON" -m pip install --no-cache-dir \
  "torch==2.11.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128

"$PYTHON" -m pip check
"$PYTHON" -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.version.cuda); print("GPU available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0)); print("Architectures:", torch.cuda.get_arch_list())'

mkdir -p runs/overnight
cp runs/real/sensors.npz runs/overnight/sensors.npz

export PINN_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$PYTHON" train.py train \
  --data-path prepared/cylinder_crop.npz \
  --run-dir runs/overnight \
  --config configs/overnight.json \
  --experiment all \
  --device cuda --resume
