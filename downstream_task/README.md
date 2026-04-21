# Downstream Task Artifact

This directory contains the downstream evaluation workflow used for checkpoints produced by the training artifacts.

## Purpose

The workflow in this directory is organized around two main steps:

1. convert a Megatron checkpoint to Hugging Face format
2. run downstream evaluation on the converted checkpoint

Accordingly, the main scripts in this directory are:

- `convert_ckpt_to_hf.sh`
- `run_llama_downstream.sh`

## Main Contents

### Conversion script
- `convert_ckpt_to_hf.sh`

### Evaluation script
- `run_llama_downstream.sh`

### Conversion utility
- `covert_llama_to_hf.py`

### Evaluation code
- `lm_eval/`

## Workflow

### Step 1: Convert checkpoint to Hugging Face format

Use:

- `convert_ckpt_to_hf.sh`

This script:

- reads the latest checkpoint iteration from `latest_checkpointed_iteration.txt`
- resolves the corresponding Megatron checkpoint file
- runs the conversion utility
- saves Hugging Face-compatible files in the checkpoint directory

### Step 2: Run downstream evaluation

Use:

- `run_llama_downstream.sh`

This script:

- reads the latest converted checkpoint
- constructs Hugging Face model arguments
- runs downstream evaluation tasks through `lm_eval`
- writes results to a timestamped results directory

## Expected Inputs

Before running, users should prepare:

- a valid checkpoint root directory
- a tokenizer directory
- a Python environment with the required packages installed
- a working `lm_eval` environment

The evaluation script assumes the converted checkpoint layout contains files such as:

- `config.json`
- `pytorch_model.bin`

under:

- `iter_XXXXXXX/mp_rank_00/`

## Main Tasks Evaluated

The provided evaluation script runs:

- `super-glue-lm-eval-v1`
- `lambada_openai`
- `lambada_standard`
- `race`
- `mathqa`
- `piqa`
- `winogrande`

## What to Edit Before Running

Please update the machine-specific fields in each script before use, such as:

- Slurm account and partition
- Python environment path
- checkpoint root path
- tokenizer path
- evaluation harness path
- cache path
- results path

## Notes on Portability

These scripts were rewritten as portable templates for artifact release.

They intentionally do **not** preserve the original private filesystem layout used during development.

## Reproduction Note

Only the main downstream conversion and evaluation workflow is highlighted here. Historical variants, machine-specific helpers, and unrelated files are not required for the intended artifact workflow.
