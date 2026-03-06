# Reconciling Communication Compression and Byzantine-Robustness in Distributed Learning

This repo supports the paper:  
**[Reconciling Communication Compression and Byzantine-Robustness in Distributed Learning](https://arxiv.org/abs/2508.17129)**, *AISTATS26*.

## Overview

This repository contains a research implementation to study the interaction between:

- **communication compression** (Rand-k sparsification with unbiased scaling), and  
- **Byzantine robustness** (coordinate-wise trimmed-mean aggregation + adversarial gradient/delta crafting)

on **MNIST** and **CIFAR-10**.

### Code layout

- `src/` contains all core components (models, data loading, compression, aggregation, clients, attacks, utilities).
- `src/train/` contains the two runnable entrypoints:
  - `src/train/mnist_train.py`
  - `src/train/cifar10_train.py`
- `scripts/` contains convenience wrappers that set `PYTHONPATH=./src` and run standard experiments.

### How an experiment runs (high level)

Both entrypoints expose the same CLI interface with the required flags:
`--kps` (compression ratio k/d), `--lr`, `--byz` (# Byzantine clients), `--algo` ∈ {`rosdhb`, `byz_dasha_page`}.


At a high level, the training loop is:

1. **Server** samples a Rand-k mask (global mask for `rosdhb`, per-client mask for `byz_dasha_page`).
2. **Honest clients** compute either:
   - a local parameter delta `Δθ_i` (for `rosdhb`) and send its compressed coordinates, or
   - a PAGE/DASHA-style message (for `byz_dasha_page`) and send its compressed coordinates.
3. **Byzantine clients (optional)** craft malicious vectors constrained to the communicated coordinates and send their packets.
4. **Server** reconstructs unbiased masked vectors (`d/k` scaling) and performs **robust aggregation** (trimmed mean).
5. **Server** updates the global model; evaluation is run periodically; results are logged and saved.



## Requirements

- Python ≥ 3.10
- PyTorch (`torch`) + `torchvision`
- `numpy`
- `matplotlib`


### Quickstart 

```bash
python -m pip install --upgrade pip
pip install torch torchvision numpy
bash scripts/run_mnist.sh
bash scripts/run_cifar10.sh 
