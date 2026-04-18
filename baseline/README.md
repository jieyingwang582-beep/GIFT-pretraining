# Baseline Artifact

This directory contains the high-precision communication baseline used in the paper.

## Purpose

This implementation serves as the reference system for comparison against:

- the Euclidean communication compression baseline
- the GIFT communication compression method

## Main Contents

### Training entry
- `pretrain_gpt.py`

### Main run scripts
- `run_scripts/run_baseline_300m.sh`
- `run_scripts/run_baseline_600m.sh`

### Code
- `megatron/`
- other modified Python files used by the baseline implementation

## Configurations Covered

### `run_scripts/run_baseline_300m.sh`
Baseline run script for the **300M** model configuration.

### `run_scripts/run_baseline_600m.sh`
Baseline run script for the **600M** model configuration.

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

## Relationship to Other Directories

- `../euclidean/` contains the direct Euclidean FP8 communication baseline
- `../GIFT/` contains the geometry-informed FP8 communication method

## Reproduction Note

Only the main paper-relevant baseline scripts are kept here. Historical variants, machine-specific helpers, and unrelated experiment scripts are intentionally omitted.
