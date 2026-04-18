#!/bin/bash
#SBATCH -J euclidean_small_64gpu
#SBATCH -A TG-NAIRR250444
#SBATCH -p gh
#SBATCH -N 64
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=72
#SBATCH -t 00:08:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jw2088@scarletmail.rutgers.edu
#SBATCH --output=/work/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_timing/slurm_logs/%x_%j.out
#SBATCH --error=/work/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_timing/slurm_logs/%x_%j.err
set -euo pipefail

# Environment setup (Same as previous baseline)
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
MEGATRON_DIR="/work/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_timing"
cd "$MEGATRON_DIR"


# Create logs dir
mkdir -p "${MEGATRON_DIR}/slurm_logs"

# Paths
TOKENIZER_PATH="/work/11070/jieyingwang8888/vista/Llama-2-Tokenizer"
DATA_PATH="/work/11070/jieyingwang8888/vista/llama_dataset/my-llama2_openwebtext_text_document"

CHECKPOINT_PATH="/scratch/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_llama_perlayer_test/checkpoints/euclidean_muon_300_test3"
TENSORBOARD_LOGS_PATH="${MEGATRON_DIR}/logs"
mkdir -p "$CHECKPOINT_PATH"
mkdir -p "$TENSORBOARD_LOGS_PATH"

# Configuration (Based on provided snippet)
GPUS_PER_NODE=1
NUM_NODES=${SLURM_NNODES}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500

WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# model arg
GPT_MODEL_ARGS=(
    --num-layers 24
    --hidden-size 1024
    --num-attention-heads 16
    --ffn-hidden-size 2736
    --seq-length 4096
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

# hyper parameters
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
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_PATH
    --tokenizer-hf-use-fast
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 1000
    --eval-interval 100
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --ckpt-format torch
)

echo "=== START $(date) ==="
echo "HOST=$(hostname)"
echo "JOBID=$SLURM_JOB_ID NNODES=$SLURM_NNODES NODELIST=$SLURM_NODELIST"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT (node_rank will be set inside srun)"
which python || true
python -c "import torch; print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())" || true

# avoid silent buffering
export PYTHONUNBUFFERED=1

# a bit more robust for NCCL startup
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "=== launching via srun: WORLD_SIZE=$WORLD_SIZE (GPUS_PER_NODE=$GPUS_PER_NODE) ==="

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
  -N "$NUM_NODES" -n "$NUM_NODES" \
  --ntasks-per-node=1 \
  --cpu-bind=none \
  bash -lc "
    set -euo pipefail
    cd '$MEGATRON_DIR'

    export TOKENIZERS_PARALLELISM=false
    export RAYON_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1

    NODE_RANK=\$SLURM_NODEID
    echo \"LAUNCH host=\$(hostname) SLURM_NODEID=\$SLURM_NODEID NODE_RANK=\$NODE_RANK\"

    # ---- caches (MUST be inside srun so SLURM_NODEID is valid) ----
    export TRITON_CACHE_DIR='/scratch/11070/jieyingwang8888/vista/cache/triton/'\"\$SLURM_JOB_ID\"'/node'\"\$SLURM_NODEID\"
    export TORCHINDUCTOR_CACHE_DIR='/scratch/11070/jieyingwang8888/vista/cache/inductor/'\"\$SLURM_JOB_ID\"'/node'\"\$SLURM_NODEID\"
    export XDG_CACHE_HOME='/scratch/11070/jieyingwang8888/vista/cache/xdg/'\"\$SLURM_JOB_ID\"'/node'\"\$SLURM_NODEID\"
    mkdir -p \"\$TRITON_CACHE_DIR\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$XDG_CACHE_HOME\"

    echo \"TRITON_CACHE_DIR=\$TRITON_CACHE_DIR\"
    echo \"TORCHINDUCTOR_CACHE_DIR=\$TORCHINDUCTOR_CACHE_DIR\"
    echo \"XDG_CACHE_HOME=\$XDG_CACHE_HOME\"

    echo \"CMD: $CMD_STR\"
    eval \"$CMD_STR\"
  "
