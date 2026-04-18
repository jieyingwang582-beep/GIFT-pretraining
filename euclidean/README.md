# Euclidean FP8 Artifact

This directory contains the **Euclidean FP8 communication baseline** used in our SC26 paper.

## Purpose

This implementation corresponds to the direct low-precision communication baseline in the paper.  
In this version, gradients are communicated in **Euclidean parameter coordinates** using FP8 communication, without the geometry-aware transform used in GIFT.

This directory is intended to serve as the comparison point between:

- the high-precision baseline in `../baseline/`
- the geometry-informed method in `../GIFT/`

## Paper-relevant model scales

In the paper, the main model scales are:

- **Llama-300M**
- **Llama-600M**

Accordingly, this directory keeps two main run scripts for these two model configurations.

## Directory contents

### Main training entry
- `pretrain_gpt.py`

### Main run scripts
- `run_scripts/run_euclidean_300m.sh`
- `run_scripts/run_euclidean_600m.sh`

### Code
- `megatron/`
- other modified Python files in this directory used by the Euclidean FP8 communication baseline

## Script descriptions

### `run_scripts/run_euclidean_300m.sh`
Main Euclidean FP8 script for the **300M** model configuration used in the paper.

### `run_scripts/run_euclidean_600m.sh`
Main Euclidean FP8 script for the **600M** model configuration used in the paper.

## Notes on portability

These scripts were rewritten as portable templates for artifact release. Before running on a new system, users should update the machine-specific fields in each script, including:

- Slurm account name
- partition name
- number of nodes / tasks
- module load commands
- Python environment path
- tokenizer path
- dataset path
- checkpoint/save path
- cache path

The scripts intentionally do **not** preserve the original private filesystem layout used during development.

## Changing GPU count

The scripts are written with default distributed settings, but users may adapt them to their own systems.

If you want to change GPU count, you will typically need to edit:

- `#SBATCH -N`
- `#SBATCH --ntasks`
- `GPUS_PER_NODE`
- possibly batch-size-related settings if your hardware differs

## Expected role in the paper

This directory provides the **Euclidean FP8 communication baseline** used in the paper’s comparisons.

It is the direct baseline against which we compare the geometry-informed communication method in `../GIFT/`.

## Relationship to the other directories

- `../baseline/` provides the high-precision communication baseline
- `../euclidean/` provides the direct Euclidean FP8 communication baseline
- `../GIFT/` provides the geometry-informed FP8 communication method

## Reproduction guidance

For artifact clarity, only the main paper-relevant Euclidean scripts are kept here. Historical variants, machine-specific launch helpers, and unrelated experiment scripts are not required for the artifact workflow.

This directory is intended to help readers and reviewers:

- inspect the Euclidean FP8 implementation
- run the 300M Euclidean FP8 configuration
- run the 600M Euclidean FP8 configuration
- compare this implementation against the high-precision and GIFT versions
