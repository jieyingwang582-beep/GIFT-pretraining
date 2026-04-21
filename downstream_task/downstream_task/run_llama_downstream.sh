#!/bin/bash
#SBATCH -J downstream
#SBATCH -A CCR26006 
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jw2088@scarletmail.rutgers.edu
#SBATCH -t 02:00:00
#SBATCH --output=/work/11070/jieyingwang8888/vista/slurm_logs/%x_%j.out
#SBATCH --error=/work/11070/jieyingwang8888/vista/slurm_logs/%x_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8

source /work/11070/jieyingwang8888/vista/downstream_env/bin/activate

export HF_HOME=/work/11070/jieyingwang8888/vista/hf_eval_cache/hf_home
export HF_DATASETS_CACHE=/work/11070/jieyingwang8888/vista/hf_eval_cache/datasets
export HF_EVALUATE_CACHE=/work/11070/jieyingwang8888/vista/hf_eval_cache/evaluate
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_EVALUATE_CACHE"

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
nvidia-smi -L || true
python -c "import torch; print('cuda', torch.cuda.is_available(), 'device_count', torch.cuda.device_count())"

CKPT_ROOT=/scratch/11070/jieyingwang8888/vista/Megatron-LM-llama-baseline_timer/checkpoints/baseline_llama_medium_64_4096
TOKENIZER_PATH="/work/11070/jieyingwang8888/vista/Llama-2-Tokenizer"

LATEST_ITER=$(tr -d '[:space:]' < "$CKPT_ROOT/latest_checkpointed_iteration.txt")
ITER_DIR=$(printf "iter_%07d" "$LATEST_ITER")
HF_CKPT_DIR="$CKPT_ROOT/$ITER_DIR/mp_rank_00"

echo "LATEST_ITER=$LATEST_ITER"
echo "HF_CKPT_DIR=$HF_CKPT_DIR"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"

test -d "$HF_CKPT_DIR"
test -d "$TOKENIZER_PATH"
test -f "$HF_CKPT_DIR/config.json"
test -f "$HF_CKPT_DIR/pytorch_model.bin"

ls -lah "$HF_CKPT_DIR" | egrep 'config.json|pytorch_model.bin|model_optim_rng.pt|tokenizer|special_tokens_map' || true
ls -lah "$TOKENIZER_PATH" || true

echo "===== config.json ====="
cat "$HF_CKPT_DIR/config.json" || true
echo "======================="

export MODEL_ARGS="pretrained=${HF_CKPT_DIR},tokenizer=${TOKENIZER_PATH},dtype=bfloat16,device_map=cuda:0"
echo "MODEL_ARGS=$MODEL_ARGS"

SEED="0,2027,2027,2027"
echo "SEED=$SEED"

RESULTS_DIR=/work/11070/jieyingwang8888/vista/lm-evaluation-harness/results/results_baseline_iter${LATEST_ITER}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"

cd /work/11070/jieyingwang8888/vista/lm-evaluation-harness

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
