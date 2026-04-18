# SC26 Artifact for GIFT

This repository contains the code artifacts for our SC26 paper:

**GIFT: Geometry-Informed Low-precision Gradient Communication for Large-Scale LLM Pretraining**

## Overview

We study low-precision gradient communication for large-scale LLM pretraining. This artifact includes three implementations used in the paper:

- `baseline/`: full-precision communication baseline
- `euclidean/`: direct Euclidean FP8 communication baseline
- `GIFT/`: our geometry-informed FP8 communication method

The three directories are intentionally separated because each one contains its own modified training entry points, Megatron code, and method-specific communication logic.

## Repository Structure

```text
sc-artifact/
├── README.md
├── README_megatron_original.md
├── baseline/
├── euclidean/
└── GIFT/

