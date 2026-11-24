#!/usr/bin/env bash
#SBATCH --job-name=cylinder-pinn
#SBATCH --output=slurm-%j.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
# Cluster account, partition, GPU request and modules are supplied at submission.
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
: "${DATA_PATH:?Set DATA_PATH to the prepared crop}"
: "${RUN_DIR:?Set RUN_DIR to an output directory}"
: "${CONFIG:?Set CONFIG to a configuration JSON}"
: "${PYTHON:?Set PYTHON to the Python executable in your environment}"
EXPERIMENT="${EXPERIMENT:-all}"
export PINN_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$RUN_DIR"
"$PYTHON" train.py verify --data-path "$DATA_PATH" --run-dir "$RUN_DIR"
"$PYTHON" train.py train --data-path "$DATA_PATH" --run-dir "$RUN_DIR" --config "$CONFIG" --experiment "$EXPERIMENT" --device cuda --resume
