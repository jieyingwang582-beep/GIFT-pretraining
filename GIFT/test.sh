#!/bin/bash
set -euo pipefail

HOLD_JOB_ID=644093
ACCOUNT=TG-NAIRR250444
PARTITION=gh
STEP_TIME=00:05:00

NODELIST=$(squeue -j "$HOLD_JOB_ID" -h -o %N)
if [ -z "$NODELIST" ]; then
  echo "Failed to get NODELIST from HOLD_JOB_ID=$HOLD_JOB_ID"
  exit 1
fi

NUM_NODES=$(scontrol show hostnames "$NODELIST" | wc -l | tr -d ' ')
MASTER_ADDR=$(scontrol show hostnames "$NODELIST" | head -n 1)
MASTER_PORT=29500
GPUS_PER_NODE=1
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

RUN_ID=$(date +%Y%m%d_%H%M%S)

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8

source /work/11070/jieyingwang8888/vista/llama_env/bin/activate

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

MEGATRON_DIR="/work/11070/jieyingwang8888/vista/Megatron-LM_IG_partial_test_timing"
cd "$MEGATRON_DIR"
mkdir -p "${MEGATRON_DIR}/slurm_logs"

LOG_OUT="${MEGATRON_DIR}/slurm_logs/IG_muon_timer_32gpu_sleep_${HOLD_JOB_ID}_${RUN_ID}.out"
LOG_ERR="${MEGATRON_DIR}/slurm_logs/IG_muon_timer_32gpu_sleep_${HOLD_JOB_ID}_${RUN_ID}.err"

unset MEGATRON_NATIVE_FP8_ALLREDUCE || true
unset MEGATRON_FP8_AR_DTYPE || true
export IG_FP8Q_ALLREDUCE=1
export IG_FP8Q_DTYPE=float8_e5m2
export IG_USE_BATCHED_SOLVE=1
export IG_BATCHED_SOLVE_MIN_BUCKET=2
export IG_COMPILE_SOLVE_MODE=default
export IG_COMPILE_MAPBACK=1
export IG_COMPILE_DYNAMIC=0
export IG_USE_INVERSE_BMM=1
export IG_USE_BATCHED_SOLVE=1
export IG_COMM_ASYNC=0
export IG_COMM_NUM_BUCKETS=1
export IG_COMPILE_SOLVE=0
export IG_LINALG_BACKEND=default

TOKENIZER_PATH="/work/11070/jieyingwang8888/vista/Llama-2-Tokenizer"
DATA_PATH="/work/11070/jieyingwang8888/vista/llama_dataset/my-llama2_openwebtext_text_document"

CHECKPOINT_PATH="/scratch/11070/jieyingwang8888/vista/Megatron-LM_IG_partial_test_timing/checkpoints/test1"
TENSORBOARD_LOGS_PATH="${MEGATRON_DIR}/logs"
mkdir -p "$CHECKPOINT_PATH"
mkdir -p "$TENSORBOARD_LOGS_PATH"

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
    --init-method-std 0.02
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
echo "HOLD_JOB_ID=$HOLD_JOB_ID"
echo "NUM_NODES=$NUM_NODES"
echo "NODELIST=$NODELIST"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "LOG_OUT=$LOG_OUT"
echo "LOG_ERR=$LOG_ERR"
which python || true
python -c "import torch; print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())" || true

export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "=== launching via srun on sleeping job: WORLD_SIZE=$WORLD_SIZE ==="

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

srun \
  --jobid "$HOLD_JOB_ID" \
  -A "$ACCOUNT" \
  -p "$PARTITION" \
  -N "$NUM_NODES" \
  -n "$NUM_NODES" \
  --ntasks-per-node=1 \
  --cpus-per-task=72 \
  --cpu-bind=none \
  -t "$STEP_TIME" \
  --output="$LOG_OUT" \
  --error="$LOG_ERR" \
  bash -lc "
    set -euo pipefail
    cd '$MEGATRON_DIR'

    source /etc/profile.d/modules.sh 2>/dev/null || true
    module purge
    module load gcc/13.2.0
    module load cuda/12.8
    module load python3/3.11.8
    source /work/11070/jieyingwang8888/vista/llama_env/bin/activate

    SITE_PACKAGES=\$(python -c 'import site; print(site.getsitepackages()[0])')
    CUDNN_HOME=\"\${SITE_PACKAGES}/nvidia/cudnn\"
    NCCL_HOME=\"\${SITE_PACKAGES}/nvidia/nccl\"
    export CUDNN_HOME
    export NCCL_HOME
    export LD_LIBRARY_PATH=\"\${CUDNN_HOME}/lib:\${NCCL_HOME}/lib:\${LD_LIBRARY_PATH:-}\"
    export CPATH=\"\${CUDNN_HOME}/include:\${NCCL_HOME}/include:\${CPATH:-}\"
    export LIBRARY_PATH=\"\${CUDNN_HOME}/lib:\${NCCL_HOME}/lib:\${LIBRARY_PATH:-}\"
    export CFLAGS=\"-I\${CUDNN_HOME}/include \${CFLAGS:-}\"
    export LDFLAGS=\"-L\${CUDNN_HOME}/lib \${LDFLAGS:-}\"

    export TOKENIZERS_PARALLELISM=false
    export RAYON_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTHONUNBUFFERED=1

    echo \"LAUNCH host=\$(hostname) SLURM_NODEID=\${SLURM_NODEID:-NA} JOBID=\${SLURM_JOB_ID:-NA}\"

    export TRITON_CACHE_DIR=/scratch/11070/jieyingwang8888/vista/cache/triton/\${SLURM_JOB_ID}/node\${SLURM_NODEID}
    export TORCHINDUCTOR_CACHE_DIR=/scratch/11070/jieyingwang8888/vista/cache/inductor/\${SLURM_JOB_ID}/node\${SLURM_NODEID}
    export XDG_CACHE_HOME=/scratch/11070/jieyingwang8888/vista/cache/xdg/\${SLURM_JOB_ID}/node\${SLURM_NODEID}
    mkdir -p \"\$TRITON_CACHE_DIR\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$XDG_CACHE_HOME\"

    unset MEGATRON_NATIVE_FP8_ALLREDUCE || true
    unset MEGATRON_FP8_AR_DTYPE || true
    export IG_FP8Q_ALLREDUCE=1
    export IG_FP8Q_DTYPE=float8_e5m2
    export IG_USE_BATCHED_SOLVE=1
    export IG_BATCHED_SOLVE_MIN_BUCKET=2
    export IG_COMPILE_SOLVE_MODE=default
    export IG_COMPILE_MAPBACK=1
    export IG_COMPILE_DYNAMIC=0
    export IG_USE_INVERSE_BMM=1
    export IG_USE_BATCHED_SOLVE=1
    export IG_COMM_ASYNC=0
    export IG_COMM_NUM_BUCKETS=1
    export IG_COMPILE_SOLVE=0
    export IG_LINALG_BACKEND=default

    echo \"TRITON_CACHE_DIR=\$TRITON_CACHE_DIR\"
    echo \"TORCHINDUCTOR_CACHE_DIR=\$TORCHINDUCTOR_CACHE_DIR\"
    echo \"XDG_CACHE_HOME=\$XDG_CACHE_HOME\"
    python -c \"import torch, os; print('CUDA_CHECK host=', os.uname().nodename, 'avail=', torch.cuda.is_available(), 'count=', torch.cuda.device_count())\"

    echo \"CMD: $CMD_STR\"
    eval \"$CMD_STR\"
  "
