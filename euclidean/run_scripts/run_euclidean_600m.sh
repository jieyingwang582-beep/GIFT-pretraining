#!/bin/bash
#SBATCH -J euclidean_600m
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_PARTITION
#SBATCH -N 32
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=72
#SBATCH -t 15:00:00
#SBATCH --output=./slurm_logs/%x_%j.out
#SBATCH --error=./slurm_logs/%x_%j.err

set -euo pipefail

###############################################################################
# Euclidean FP8 600M run script
#
# This script corresponds to the Euclidean FP8 communication baseline.
# Default setting: 32 nodes, 1 GPU per node.
#
# Before running on a new system, please update:
#   1. SBATCH account / partition / node count / time limit
#   2. module load commands
#   3. VENV_PATH
#   4. TOKENIZER_PATH
#   5. DATA_PATH
#   6. CHECKPOINT_PATH
#   7. CACHE_ROOT
#
# If you change GPU count, you may need to update:
#   - #SBATCH -N
#   - #SBATCH --ntasks
#   - GPUS_PER_NODE
#   - batch-size-related settings if desired
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
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-/path/to/your/venv}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/tokenizer}"
DATA_PATH="${DATA_PATH:-/path/to/dataset_prefix}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_DIR}/outputs/euclidean_600m_checkpoints}"
TENSORBOARD_LOGS_PATH="${TENSORBOARD_LOGS_PATH:-${REPO_DIR}/outputs/tensorboard/euclidean_600m}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_DIR}/outputs/cache}"

cd "$REPO_DIR"

mkdir -p "${REPO_DIR}/slurm_logs"
mkdir -p "$CHECKPOINT_PATH"
mkdir -p "$TENSORBOARD_LOGS_PATH"
mkdir -p "$CACHE_ROOT"

# -------------------------
# Python environment
# -------------------------
source "${VENV_PATH}/bin/activate"

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
CUDNN_HOME="${SITE_PACKAGES}/nvidia/cudnn"
NCCL_HOME="${SITE_PACKAGES}/nvidia/nccl"

export CUDNN_HOME
export NCCL_HOME
export LD_LIBRARY_PATH="${CUDNN_HOME}/lib:${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${CUDNN_HOME}/include:${NCCL_HOME}/include:${CPATH:-}"
export LIBRARY_PATH="${CUDNN_HOME}/lib:${NCCL_HOME}/lib:${LIBRARY_PATH:-}"
export CFLAGS="-I${CUDNN_HOME}/include ${CFLAGS:-}"
export LDFLAGS="-L${CUDNN_HOME}/lib ${LDFLAGS:-}"

export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export FP8_MLP_BOUNDARY_STEP_LOG=0
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# -------------------------
# Distributed launch config
# -------------------------
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NUM_NODES="${SLURM_NNODES:-1}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

# -------------------------
# Model configuration (600M)
# -------------------------
GPT_MODEL_ARGS=(
    --num-layers 32
    --hidden-size 1280
    --num-attention-heads 20
    --ffn-hidden-size 3456
    --seq-length 2048
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-base 10000
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --disable-bias-linear
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --init-method-std 0.02
    --use-flash-attn
)

TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 512
    --train-iters 60000
    --weight-decay 0.1
    --optimizer muon
    --muon-momentum 0.9
    --muon-extra-scale-factor 0.2
    --muon-scale-mode spectral
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --bf16
    --lr 5e-4
    --lr-decay-style cosine
    --min-lr 5e-5
    --lr-warmup-iters 6000
    --lr-decay-iters 60000
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_PATH"
    --tokenizer-hf-use-fast
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 1000
    --eval-interval 100
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --eval-iters 10
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --ckpt-format torch
)

echo "=== START $(date) ==="
echo "HOST=$(hostname)"
echo "JOBID=${SLURM_JOB_ID:-none} NNODES=$NUM_NODES NODELIST=${SLURM_NODELIST:-none}"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "REPO_DIR=$REPO_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "WORLD_SIZE=$WORLD_SIZE"

which python || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())" || true

CMD=(torchrun
  --nproc_per_node "$GPUS_PER_NODE"
  --nnodes "$NUM_NODES"
  --node_rank "\$SLURM_NODEID"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  pretrain_gpt.py
  "${GPT_MODEL_ARGS[@]}"
  "${TRAINING_ARGS[@]}"
  "${MODEL_PARALLEL_ARGS[@]}"
  "${DATA_ARGS[@]}"
  "${EVAL_AND_LOGGING_ARGS[@]}"
)

CMD_STR=$(printf '%q ' "${CMD[@]}")

echo "=== launching via srun: WORLD_SIZE=$WORLD_SIZE (GPUS_PER_NODE=$GPUS_PER_NODE) ==="

srun \
  -N "$NUM_NODES" -n "$NUM_NODES" \
  --ntasks-per-node=1 \
  --cpu-bind=none \
  bash -lc "
    set -euo pipefail
    cd '$REPO_DIR'

    export TOKENIZERS_PARALLELISM=false
    export RAYON_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export FP8_MLP_BOUNDARY_STEP_LOG=0
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1

    NODE_RANK=\$SLURM_NODEID
    echo \"LAUNCH host=\$(hostname) SLURM_NODEID=\$SLURM_NODEID NODE_RANK=\$NODE_RANK\"

    export TRITON_CACHE_DIR='$CACHE_ROOT/triton/'\"\$SLURM_JOB_ID\"'/node'\"\$SLURM_NODEID\"
    export TORCHINDUCTOR_CACHE_DIR='$CACHE_ROOT/inductor/'\"\$SLURM_JOB_ID\"'/node'\"\$SLURM_NODEID\"
    export XDG_CACHE_HOME='$CACHE_ROOT/xdg/'\"\$SLURM_JOB_ID\"'/node'\"\$SLURM_NODEID\"
    mkdir -p \"\$TRITON_CACHE_DIR\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$XDG_CACHE_HOME\"

    echo \"TRITON_CACHE_DIR=\$TRITON_CACHE_DIR\"
    echo \"TORCHINDUCTOR_CACHE_DIR=\$TORCHINDUCTOR_CACHE_DIR\"
    echo \"XDG_CACHE_HOME=\$XDG_CACHE_HOME\"

    echo \"CMD: $CMD_STR\"
    eval \"$CMD_STR\"
  "
