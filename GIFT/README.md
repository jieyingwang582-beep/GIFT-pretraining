# GIFT Artifact

This directory contains the implementation of **GIFT**, the geometry-informed communication method used in the paper.

## Main Contents

### Training entry
- `pretrain_gpt.py`

### Main run scripts
- `run_scripts/run_gift_300m.sh`
- `run_scripts/run_gift_600m.sh`

### Code
- `megatron/`
- other modified Python files used by the GIFT communication path

## Configurations Covered

### `run_scripts/run_gift_300m.sh`
GIFT run script for the **300M** model configuration.

### `run_scripts/run_gift_600m.sh`
GIFT run script for the **600M** model configuration.

## Method Summary

GIFT performs low-precision gradient communication in **geometry-aware coordinates** rather than directly in Euclidean parameter coordinates.

The practical design used here follows three main simplifications:

- retain only the input-side geometry
- approximate the geometry using a low-rank representation
- apply the geometry-aware branch selectively to the most vulnerable layers

Most layers still use the standard communication path, while selected layers use the geometry-informed transform.

## Method-Specific Environment

The provided GIFT scripts use the following method-specific environment settings:

- `IG_FP8Q_ALLREDUCE=1`
- `IG_FP8Q_DTYPE=float8_e5m2`
- `IG_USE_BATCHED_SOLVE=1`
- `IG_BATCHED_SOLVE_MIN_BUCKET=2`
- `IG_COMPILE_SOLVE_MODE=default`
- `IG_COMPILE_MAPBACK=1`
- `IG_COMPILE_DYNAMIC=0`
- `IG_USE_INVERSE_BMM=1`
- `IG_COMM_ASYNC=0`
- `IG_COMM_NUM_BUCKETS=1`
- `IG_COMPILE_SOLVE=0`
- `IG_LINALG_BACKEND=default`

The scripts also explicitly unset:

- `MEGATRON_NATIVE_FP8_ALLREDUCE`
- `MEGATRON_FP8_AR_DTYPE`

before enabling the GIFT communication path.

## What to Edit Before Running

Please update the machine-specific fields in each script before use, such as:

- Slurm account and partition
- node count and task count
- Python environment path
- tokenizer path
- dataset path
- checkpoint path
- cache path

## Changing GPU Count

If you want to change GPU count, you will typically need to edit:

- `#SBATCH -N`
- `#SBATCH --ntasks`
- `GPUS_PER_NODE`

You may also need to adjust batch-related settings depending on your hardware.

## Reproduction Note

Only the main paper-relevant GIFT scripts are kept here. Historical variants, machine-specific helpers, and unrelated experiment scripts are intentionally omitted.
