#!/bin/bash
#SBATCH -J debug_euclidean
#SBATCH -A CCR26006
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH -t 00:06:00
#SBATCH --output=/work/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_2/slurm_logs/debug_%j.out
#SBATCH --error=/work/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_2/slurm_logs/debug_%j.err

# === 0. 设置调试日志文件 (所有输出都会同时存到这里) ===
LOG_FILE="/work/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_2/debug_crash_report_${SLURM_JOB_ID}.txt"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "   STARTING DEEP DEBUG MODE at $(date)"
echo "   Log saved to: $LOG_FILE"
echo "=============================================="

set -u # 遇到未定义变量报错
# set -e # 暂时关闭 -e，防止中间的调试命令失败导致脚本直接退出，看不到后面的日志

# === 1. 环境加载 ===
echo "[DEBUG] Loading Modules..."
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8

source /work/11070/jieyingwang8888/vista/Megatron_env/py311_cu128/bin/activate

# 环境变量设置
export CUDNN_HOME=/work/11070/jieyingwang8888/vista/third_party/cudnn
export CPATH="$CUDNN_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDNN_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"

MEGATRON_DIR="/work/11070/jieyingwang8888/vista/Megatron-LM-Euclidean_2"
cd "$MEGATRON_DIR" || { echo "CRITICAL: Cannot cd into $MEGATRON_DIR"; exit 1; }

echo "[DEBUG] Current Directory: $(pwd)"
echo "[DEBUG] Python Path: $(which python)"

# === 2. 关键变量处理 (防止 KeyError) ===
echo "[DEBUG] Handling Environment Variables..."
# 不要 unset，改为设为空字符串或 '0'，防止 Python os.environ['KEY'] 报错
export MEGATRON_NATIVE_FP8_ALLREDUCE=""
export MEGATRON_FP8_AR_DTYPE=""
export IG_FP8Q_ALLREDUCE=""
export IG_FP8Q_DTYPE=""

# 强制开启 Python 详细输出
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# === 3. 准备参数 (使用扁平化写法，防止隐形字符) ===
DATA_PATH="/work/11070/jieyingwang8888/vista/Megatron_dataset/my-gpt2_text_document"
VOCAB_FILE="/work/11070/jieyingwang8888/vista/Megatron_dataset/gpt2-vocab.json"
MERGE_FILE="/work/11070/jieyingwang8888/vista/Megatron_dataset/gpt2-merges.txt"
CHECKPOINT_PATH="${MEGATRON_DIR}/checkpoints/debug_run"
mkdir -p "$CHECKPOINT_PATH"

# 这里手动拼接所有参数，确保没有换行符或数组问题
GPT_ARGS="--num-layers 12 --hidden-size 768 --num-attention-heads 12 --seq-length 1024 --max-position-embeddings 1024 --attention-backend auto"
TRAIN_ARGS="--micro-batch-size 16 --global-batch-size 512 --train-iters 100 --bf16 --lr 2e-4 --min-lr 2e-5 --lr-warmup-iters 10 --lr-decay-style cosine --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.999 --init-method-std 0.02"
PARALLEL_ARGS="--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1"
DATA_ARGS="--data-path $DATA_PATH --tokenizer-type GPT2BPETokenizer --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE --split 949,50,1"
OUTPUT_ARGS="--log-interval 1 --save-interval 50 --eval-interval 50 --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH --eval-iters 10"

ALL_ARGS="$GPT_ARGS $TRAIN_ARGS $PARALLEL_ARGS $DATA_ARGS $OUTPUT_ARGS"

# === 4. 只有这一步是关键：先进行单机语法检查！ ===
echo "=============================================="
echo "   STEP 1: SYNTAX & ARGUMENT CHECK (Dry Run)"
echo "   Running python pretrain_gpt.py --help to catch syntax errors..."
echo "=============================================="

# 如果这里报错，说明代码本身有语法错误，或者某个 import 找不到
python pretrain_gpt.py --help > /dev/null
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "CRITICAL FAILURE: pretrain_gpt.py cannot even run --help!"
    echo "This means there is a SYNTAX ERROR or MISSING LIBRARY."
    echo "Running again without capturing output to show you the error:"
    echo "----------------------------------------------"
    python pretrain_gpt.py --help
    echo "----------------------------------------------"
    echo "Stopping script here."
    exit 1
else
    echo "[PASS] pretrain_gpt.py syntax seems OK."
fi

# === 5. 启动分布式训练 (带详细日志捕捉) ===
GPUS_PER_NODE=1 # 保持和 SBATCH 一致
NUM_NODES=${SLURM_NNODES}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29501 # 换个端口防止冲突

echo "=============================================="
echo "   STEP 2: LAUNCHING TORCHRUN"
echo "   MASTER_ADDR=$MASTER_ADDR : $MASTER_PORT"
echo "   WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))"
echo "=============================================="

srun -p gh-dev -A CCR26006 \
  -N $NUM_NODES -n $(($GPUS_PER_NODE*$NUM_NODES)) \
  --ntasks-per-node=$GPUS_PER_NODE \
  --cpu-bind=none \
  bash -lc "
    set -u
    export NODE_RANK=\$SLURM_PROCID
    echo \"[SRUN] Launching Rank \$NODE_RANK on \$(hostname)\"
    
    # 打印一下实际运行的完整命令供检查
    # echo \"Command: torchrun ... pretrain_gpt.py $ALL_ARGS\"
    
    torchrun \
      --nproc_per_node=$GPUS_PER_NODE \
      --nnodes=$NUM_NODES \
      --node_rank=\$NODE_RANK \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      pretrain_gpt.py \
      $ALL_ARGS
  "

RUN_EXIT_CODE=$?

echo "=============================================="
if [ $RUN_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Job finished successfully."
else
    echo "FAILURE: Job exited with code $RUN_EXIT_CODE"
    echo "PLEASE CHECK THE FILE: $LOG_FILE"
fi
echo "=============================================="
