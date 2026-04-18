# Euclidean FP8 Artifact

This directory contains the direct FP8 communication implementation in Euclidean parameter coordinates.

## Main Contents

### Training entry
- `pretrain_gpt.py`

### Main run scripts
- `run_scripts/run_euclidean_300m.sh`
- `run_scripts/run_euclidean_600m.sh`

### Code
- `megatron/`
- other modified Python files used by the Euclidean FP8 communication path

## Configurations Covered

### `run_scripts/run_euclidean_300m.sh`
Euclidean FP8 run script for the **300M** model configuration.

### `run_scripts/run_euclidean_600m.sh`
Euclidean FP8 run script for the **600M** model configuration.

## Method-Specific Environment

The Euclidean FP8 implementation does not require the geometry-aware environment settings used in the `GIFT/` directory.

In the provided scripts, the main method-specific runtime setting is:

- `FP8_MLP_BOUNDARY_STEP_LOG=0`

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

Only the main paper-relevant Euclidean scripts are kept here. Historical variants, machine-specific helpers, and unrelated experiment scripts are intentionally omitted.
