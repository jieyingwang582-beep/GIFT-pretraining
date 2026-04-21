#!/bin/bash
#SBATCH -J downstream_eval
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_PARTITION
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH --output=./slurm_logs/%x_%j.out
#SBATCH --error=./slurm_logs/%x_%j.err

set -euo pipefail

###############################################################################
# Run downstream evaluation for a Hugging Face checkpoint converted from
# a Megatron checkpoint.
#
# Before running on a new system, please update:
#   1. SBATCH account / partition / time limit
#   2. module load commands
#   3. VENV_PATH
#   4. CKPT_ROOT
#   5. TOKENIZER_PATH
#   6. HARNESS_DIR
#   7. HF cache paths if desired
#
# This script assumes:
#   - latest_checkpointed_iteration.txt exists under CKPT_ROOT
#   - the converted checkpoint layout contains:
#       iter_XXXXXXX/mp_rank_00/config.json
#       iter_XXXXXXX/mp_rank_00/pytorch_model.bin
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
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/tokenizer}"
HARNESS_DIR="${HARNESS_DIR:-${REPO_DIR}}"
RESULTS_ROOT="${RESULTS_ROOT:-${HARNESS_DIR}/results}"
CACHE_ROOT="${CACHE_ROOT:-${HARNESS_DIR}/hf_eval_cache}"

mkdir -p "${REPO_DIR}/slurm_logs"
mkdir -p "$RESULTS_ROOT"
mkdir -p "$CACHE_ROOT"

# -------------------------
# Python environment
# -------------------------
source "${VENV_PATH}/bin/activate"

# -------------------------
# Hugging Face cache
# -------------------------
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-${CACHE_ROOT}/evaluate}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_EVALUATE_CACHE"

# -------------------------
# Device setup
# -------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

nvidia-smi -L || true
python -c "import torch; print('cuda', torch.cuda.is_available(), 'device_count', torch.cuda.device_count())" || true

# -------------------------
# Resolve latest converted checkpoint
# -------------------------
LATEST_FILE="${CKPT_ROOT}/latest_checkpointed_iteration.txt"

if [[ ! -f "$LATEST_FILE" ]]; then
    echo "ERROR: latest checkpoint marker not found: $LATEST_FILE"
    exit 1
fi

LATEST_ITER=$(tr -d '[:space:]' < "$LATEST_FILE")
ITER_DIR=$(printf "iter_%07d" "$LATEST_ITER")
HF_CKPT_DIR="${CKPT_ROOT}/${ITER_DIR}/mp_rank_00"

echo "LATEST_ITER=$LATEST_ITER"
echo "HF_CKPT_DIR=$HF_CKPT_DIR"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"

test -d "$HF_CKPT_DIR"
test -d "$TOKENIZER_PATH"
test -f "$HF_CKPT_DIR/config.json"
test -f "$HF_CKPT_DIR/pytorch_model.bin"

ls -lah "$HF_CKPT_DIR" | egrep 'config.json|pytorch_model.bin|model_optim_rng.pt|tokenizer|special_tokens_map|model.safetensors' || true
ls -lah "$TOKENIZER_PATH" || true

echo "===== config.json ====="
cat "$HF_CKPT_DIR/config.json" || true
echo "======================="

# -------------------------
# Evaluation config
# -------------------------
export MODEL_ARGS="pretrained=${HF_CKPT_DIR},tokenizer=${TOKENIZER_PATH},dtype=bfloat16,device_map=cuda:0"
echo "MODEL_ARGS=$MODEL_ARGS"

SEED="${SEED:-0,2027,2027,2027}"
echo "SEED=$SEED"

RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/results_iter${LATEST_ITER}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"

cd "$HARNESS_DIR"

run_one () {
  local task="$1"
  echo "=============================="
  echo "RUN task=$task"
  echo "=============================="
  lm_eval --model hf \
    --model_args "$MODEL_ARGS" \
    --tasks "$task" \
    --device cuda:0 \
    --batch_size 1 \
    --seed "$SEED" \
    --output_path "$RESULTS_DIR/$task"
}

run_one super-glue-lm-eval-v1
run_one lambada_openai
run_one lambada_standard
run_one race
run_one mathqa
run_one piqa
run_one winogrande

echo "ALL DONE. Results in: $RESULTS_DIR"
ls -lah "$RESULTS_DIR" | head
