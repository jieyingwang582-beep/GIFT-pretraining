#!/bin/bash
#SBATCH -J ckpt_to_hf
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_PARTITION
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:05:00
#SBATCH --output=./slurm_logs/%x_%j.out
#SBATCH --error=./slurm_logs/%x_%j.err

set -euo pipefail

###############################################################################
# Convert a Megatron checkpoint to Hugging Face format
#
# Before running on a new system, please update:
#   1. SBATCH account / partition / time limit
#   2. module load commands
#   3. VENV_PATH
#   4. CKPT_ROOT
#   5. CONVERT_SCRIPT
#
# This script assumes:
#   - latest_checkpointed_iteration.txt exists under CKPT_ROOT
#   - the checkpoint layout contains:
#       iter_XXXXXXX/mp_rank_00/model_optim_rng.pt
###############################################################################

# -------------------------
# Optional environment setup
# -------------------------
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8

# -------------------------
# User-editable paths
# -------------------------
REPO_DIR="${REPO_DIR:-$(pwd)}"
VENV_PATH="${VENV_PATH:-/path/to/your/venv}"
CKPT_ROOT="${CKPT_ROOT:-/path/to/checkpoint_root}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-${REPO_DIR}/convert_llama_to_hf.py}"

mkdir -p "${REPO_DIR}/slurm_logs"

# -------------------------
# Python environment
# -------------------------
source "${VENV_PATH}/bin/activate"

# -------------------------
# Resolve latest checkpoint
# -------------------------
LATEST_FILE="${CKPT_ROOT}/latest_checkpointed_iteration.txt"

if [[ ! -f "$LATEST_FILE" ]]; then
    echo "ERROR: latest checkpoint marker not found: $LATEST_FILE"
    exit 1
fi

LATEST_ITER=$(tr -d '[:space:]' < "$LATEST_FILE")
ITER_DIR=$(printf "iter_%07d" "$LATEST_ITER")
CKPT_FILE="${CKPT_ROOT}/${ITER_DIR}/mp_rank_00/model_optim_rng.pt"

if [[ ! -f "$CKPT_FILE" ]]; then
    echo "ERROR: checkpoint file not found: $CKPT_FILE"
    exit 1
fi

if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    echo "ERROR: conversion script not found: $CONVERT_SCRIPT"
    exit 1
fi

echo "REPO_DIR=$REPO_DIR"
echo "CKPT_ROOT=$CKPT_ROOT"
echo "LATEST_ITER=$LATEST_ITER"
echo "CKPT_FILE=$CKPT_FILE"
echo "CONVERT_SCRIPT=$CONVERT_SCRIPT"

python "$CONVERT_SCRIPT" "$CKPT_FILE"

echo "DONE. Saved to $(dirname "$CKPT_FILE")"
ls -lah "$(dirname "$CKPT_FILE")" | egrep 'config.json|pytorch_model.bin|model_optim_rng.pt|model.safetensors' || true
