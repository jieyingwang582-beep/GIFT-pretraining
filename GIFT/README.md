# GIFT Artifact

This directory contains the implementation of **GIFT**, the geometry-informed FP8 communication method used in our paper.

## Purpose

GIFT is the main method proposed in the paper.  
Its key idea is to perform low-precision gradient communication in **geometry-aware coordinates** rather than directly in Euclidean parameter coordinates.

Compared with the Euclidean FP8 baseline, GIFT introduces geometry-aware communication only where it is most beneficial.

## Paper-relevant model scales

In the paper, the main model scales are:

- **Llama-300M**
- **Llama-600M**

Accordingly, this directory keeps two main run scripts for these two model configurations.

## Directory contents

### Main training entry
- `pretrain_gpt.py`

### Main run scripts
- `run_scripts/run_gift_300m.sh`
- `run_scripts/run_gift_600m.sh`

### Code
- `megatron/`
- other modified Python files in this directory used by the GIFT communication path

## Script descriptions

### `run_scripts/run_gift_300m.sh`
Main GIFT script for the **300M** model configuration used in the paper.

### `run_scripts/run_gift_600m.sh`
Main GIFT script for the **600M** model configuration used in the paper.

## Method summary

The practical GIFT design used in the paper follows three main simplifications:

- retain only the input-side geometry
- approximate the geometry using a low-rank representation
- apply the geometry-aware branch selectively to the most vulnerable layers

Most layers still use the standard Euclidean communication path, while selected layers use the geometry-informed transform.

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

## Relationship to the other directories

- `../baseline/` provides the high-precision communication baseline
- `../euclidean/` provides the direct Euclidean FP8 communication baseline
- `../GIFT/` provides the geometry-informed FP8 communication method

## Expected role in the paper

This directory provides the implementation of the paper’s main method.

It is intended to support comparison against:

- the high-precision baseline in `../baseline/`
- the Euclidean FP8 baseline in `../euclidean/`

## Reproduction guidance

For artifact clarity, only the main paper-relevant GIFT scripts are kept here. Historical variants, machine-specific launch helpers, and unrelated experiment scripts are not required for the artifact workflow.

This directory is intended to help readers and reviewers:

- inspect the GIFT implementation
- run the 300M GIFT configuration
- run the 600M GIFT configuration
- compare GIFT against the baseline and Euclidean FP8 versions
