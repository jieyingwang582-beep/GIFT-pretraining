# Baseline Artifact

This directory contains the baseline training implementation used in our paper.

## Purpose

This baseline corresponds to the high-precision communication reference used for comparison against:

- the Euclidean FP8 communication baseline
- the GIFT geometry-informed FP8 communication method

In the paper, the main model scales are:

- **Llama-300M**
- **Llama-600M**

Accordingly, this directory keeps two main run scripts for these two model configurations.

## Directory contents

### Main training entry
- `pretrain_gpt.py`

### Main run scripts
- `run_scripts/run_baseline_300m.sh`
- `run_scripts/run_baseline_600m.sh`

### Code
- `megatron/`
- other modified Python files in this directory, such as model/config/provider code used by the baseline

## Script descriptions

### `run_scripts/run_baseline_300m.sh`
Main baseline script for the **300M** model configuration used in the paper.

### `run_scripts/run_baseline_600m.sh`
Main baseline script for the **600M** model configuration used in the paper.

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

This directory provides the baseline reference implementation corresponding to the high-precision communication setting used in the paper’s comparisons.

For the other methods, see:

- `../euclidean/`
- `../GIFT/`

## Reproduction guidance

For artifact clarity, only the main paper-relevant baseline scripts are kept here. Historical variants, machine-specific launch helpers, and unrelated experiment scripts are not required for the artifact workflow.

This directory is intended to help readers and reviewers:

- inspect the baseline implementation
- run the baseline 300M configuration
- run the baseline 600M configuration
- compare this baseline against the Euclidean and GIFT implementations
