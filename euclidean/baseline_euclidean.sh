#!/bin/bash
#SBATCH -J baseline_euclidean
#SBATCH -A CCR26006
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH -t 15:00:00
#SBATCH --output=/work/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_2/slurm_logs/%x_%j.out
#SBATCH --error=/work/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_2/slurm_logs/%x_%j.err
set -euo pipefail

# Environment setup (Same as previous baseline)
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8

source /work/11070/jieyingwang8888/vista/Megatron_env/py311_cu128/bin/activate
export CUDNN_HOME=/work/11070/jieyingwang8888/vista/third_party/cudnn
export CPATH="$CUDNN_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDNN_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"
export CFLAGS="-I$CUDNN_HOME/include ${CFLAGS:-}"
export LDFLAGS="-L$CUDNN_HOME/lib ${LDFLAGS:-}"


MEGATRON_DIR="/work/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_2"
cd "$MEGATRON_DIR"

export FP8_BASELINE_LOG_EVERY=1
echo "== RUN (4GPU, 125M, bf16, global=512, micro=8) [FP8 BASELINE] =="
# Create logs dir
mkdir -p "${MEGATRON_DIR}/slurm_logs"

# Paths (Pointing to existing data/vocab from the original setup)
OLD_DIR="/work/11070/jieyingwang8888/vista/Megatron_dataset"
VOCAB_FILE="${OLD_DIR}/gpt2-vocab.json"
MERGE_FILE="${OLD_DIR}/gpt2-merges.txt"
DATA_PATH="/work/11070/jieyingwang8888/vista/Megatron_dataset/my-gpt2_text_document"

CHECKPOINT_PATH="/scratch/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_2/checkpoints/baseline_euclidean"
TENSORBOARD_LOGS_PATH="${MEGATRON_DIR}/logs"
mkdir -p "$CHECKPOINT_PATH"
mkdir -p "$TENSORBOARD_LOGS_PATH"

# Configuration (Based on provided snippet)
GPUS_PER_NODE=1  # Adapted to 4 for this environment
NUM_NODES=${SLURM_NNODES}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

# model arg
GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 768
    --num-attention-heads 12
    --seq-length 1024
    --max-position-embeddings 1024
    --attention-backend auto
)

# hyper parameters
TRAINING_ARGS=(
    --micro-batch-size 32
    --global-batch-size 512
    --train-iters 100000
    --weight-decay 0.1
    --adam-beta1 0.9 
    --adam-beta2 0.999
    --init-method-std 0.02
    --clip-grad 1.0 
    --bf16
    --lr 2e-4
    --lr-decay-style cosine 
    --min-lr 2e-5
    --lr-warmup-iters 2000
    --lr-decay-iters 100000  
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
    --legacy-tokenizer
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
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "=== launching via srun: WORLD_SIZE=$WORLD_SIZE (GPUS_PER_NODE=$GPUS_PER_NODE) ==="

srun -p gh-dev -A CCR26006 \
  -N $NUM_NODES -n $WORLD_SIZE \
  --ntasks-per-node=$GPUS_PER_NODE \
  --cpu-bind=none \
  bash -lc '
    set -euo pipefail

    export NODE_RANK=$SLURM_PROCID
    echo "LAUNCH host=$(hostname) SLURM_PROCID=$SLURM_PROCID NODE_RANK=$NODE_RANK"

    torchrun \
      --nproc_per_node='"$GPUS_PER_NODE"' \
      --nnodes='"$NUM_NODES"' \
      --node_rank=$NODE_RANK \
      --master_addr='"$MASTER_ADDR"' \
      --master_port='"$MASTER_PORT"' \
      pretrain_gpt.py \
        '"${GPT_MODEL_ARGS[*]}"' \
        '"${TRAINING_ARGS[*]}"' \
        '"${MODEL_PARALLEL_ARGS[*]}"' \
        '"${DATA_ARGS[*]}"' \
        '"${EVAL_AND_LOGGING_ARGS[*]}"'
  '

