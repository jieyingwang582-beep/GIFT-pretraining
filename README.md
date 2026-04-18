# Artifact for GIFT

This repository contains the code artifacts for our paper:

**GIFT: Geometry-Informed Low-precision Gradient Communication for Large-Scale LLM Pretraining**

## Overview

We study low-precision gradient communication for large-scale LLM pretraining. This artifact includes three implementations used in the paper:

- `baseline/`: full-precision communication baseline
- `euclidean/`: direct Euclidean FP8 communication baseline
- `GIFT/`: our geometry-informed FP8 communication method

The three directories are intentionally separated because each one contains its own modified training entry points, Megatron code, and method-specific communication logic.

## Repository Structure

```text
README.md
README_megatron_original.md
baseline/
euclidean/
GIFT/
```

## Shared Environment

The implementations in this repository were developed and tested in a Linux HPC environment with Slurm-based job launch.

The run scripts assume a software stack based on:

- GCC 13.2.0
- CUDA 12.8
- Python 3.11.8
- PyTorch installed inside a Python virtual environment
- multi-node launch through `srun` and `torchrun`

The default launch style in the provided scripts assumes:

- 1 GPU per node
- Slurm-managed node allocation
- one launcher task per node
- NCCL-based distributed communication

## Shared Training Scope

This repository is organized around two Llama-style model configurations:

- 300M: sequence length 4096
- 600M: sequence length 2048

Across the provided scripts, the main shared training settings include:

- global batch size 512
- micro batch size 4
- 60,000 training iterations
- Muon optimizer
- BF16 training
- cosine learning-rate decay
- learning rate `5e-4`
- minimum learning rate `5e-5`
- warmup over 6,000 iterations
- gradient clipping at 1.0

## Data and Tokenization

The scripts assume:

- a Hugging Face tokenizer
- a tokenizer model path provided by the user
- a dataset path provided as a Megatron-style indexed dataset prefix
- dataset split `949,50,1`

## Portability

The original development environment used machine-specific paths, Slurm account settings, and cluster-specific cache and output locations. In this artifact, the run scripts were rewritten as portable templates.

Before running on a new system, users should update:

- Slurm account and partition
- node count and job time
- Python environment path
- tokenizer path
- dataset path
- checkpoint path
- cache path
- any site-specific module load commands

## Method-Specific Instructions

Please see the README file inside each subdirectory for method-specific details:

- `baseline/README.md`
- `euclidean/README.md`
- `GIFT/README.md`

In particular:

- `baseline/` provides the high-precision communication reference
- `euclidean/` provides the direct FP8 communication baseline in Euclidean parameter coordinates
- `GIFT/` provides the geometry-informed FP8 communication method
