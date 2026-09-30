# FLAIR-pLM Development Playground

This repository is the shared workspace for refactoring and consolidating the FLAIR-pLM codebases.

## Table of Contents

- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Create the environment and install dependencies](#2-create-the-environment-and-install-dependencies)
  - [3. Activate the environment](#3-activate-the-environment)
  - [4. (Optional) Install FlashAttention 4](#4-optional-install-flashattention-4)
  - [5. Install external binaries](#5-install-external-binaries)
  - [6. Note on multi-node / shared-filesystem clusters](#6-note-on-multi-node--shared-filesystem-clusters)
- [Setup Pre-Commits For Local Development](#setup-pre-commits-for-local-development)
- [Modules](#modules)
- [Scope](#scope)

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/): Python environment and package manager.
- `git` (typically pre-installed on Slurm clusters).
- `wget`,`gcc`/`make`, and `zlib`: required to install and compile binaries such as `mmseqs2`, `seqkit` and `pigz` for the data pipeline (typically bundled into the default standard environments on Slurm clusters).

### 1. Clone the repository

```sh
git clone https://github.com/milatechtransfer/flair-plm.git
cd flair-plm
```

### 2. Create the environment and install dependencies

Running `uv sync` creates the virtual environment (`.venv/`) and installs the locked dependencies in a single step:

```sh
uv sync --extra all
```

Note: `--extra all` installs the dependencies for every non-hardware-specific module. To set up a single module instead, replace `all` with one of: `data`, `pretrain`, `evaluate`, `dev`. Add `--extra compute` only on Linux CUDA 13 systems where FlashAttention 4 is needed.

The repository's `.python-version` pins the Python version, so `uv sync` uses that interpreter automatically when it is available.

### 3. Activate the environment

```sh
source .venv/bin/activate
```

Alternatively, prefix commands with `uv run` (e.g. `uv run python ...`) to run them without activating.

### 4. (Optional) Install FlashAttention 4

Packed pretraining works with PyTorch's built-in variable-length attention and does **not** require FlashAttention. Install FlashAttention 4 to use its faster Hopper/Blackwell kernel. On supported Linux Python versions, the `compute` extra installs the CUDA 13 FA4 package through the normal dependency resolver so the PyTorch and FA4 requirements are solved together:

```sh
uv sync --extra all --extra compute
```

For editable installs, use:

```sh
uv pip install -e ".[compute]"
```

`--extra all` intentionally does **not** include FA4 because it is CUDA-specific and not needed for most environments.

Pretraining automatically uses FA4 on Hopper and newer GPUs (compute capability 9.0+) when its `flash_attn.cute` kernel is installed. Older hardware, or environments without FA4 installed, use PyTorch variable-length attention.

### 5. Install external binaries

This downloads/compiles `mmseqs2`, `seqkit`, `pigz`, and `pv` into `.venv/bin/`. It requires the virtual environment to already exist (run step 2 first):

```sh
bash scripts/setup_binaries.sh
```

### 6. Note on multi-node / shared-filesystem clusters

By default, uv creates the environment as `.venv/` **inside the project directory**. If the repository lives on a **shared filesystem** and you run the pipeline from **multiple nodes** against the same checkout, this is unsafe: when a node's `uv run`/`uv sync` resolves a different interpreter than the one that built the shared `.venv` (e.g. a freshly installed uv falling back to the system `python3`), uv will **remove and recreate** `.venv`, wiping the environment and the compiled binaries from step 5 out from under the other nodes.

To avoid this, you can give each node its **own** environment on **node-local** storage by setting `UV_PROJECT_ENVIRONMENT` before running any `uv` command:

```sh
export UV_PROJECT_ENVIRONMENT=/tmp/flair-plm-venv   # node-local path (NOT on the shared filesystem)
export UV_LINK_MODE=copy                            # avoids hardlink warnings across filesystems

uv sync --extra all
bash scripts/setup_binaries.sh                      # binaries follow UV_PROJECT_ENVIRONMENT automatically
```

Run these once per node. Because each node now owns its environment, an interpreter mismatch only rebuilds that node's own venv and can never clobber another node's. Export the same two variables in your job/launch script so every `uv run` uses the node-local environment.

## Setup Pre-Commits For Local Development

Ensure that you have sourced the environment before running the following commands.

1. Ensure that developer dependencies are installed:

```sh
uv sync --extra dev
```

2. Install hooks:

```sh
make setup-pre-commit
```

3. Setup Github
```sh
make setup-github
```

4. Run checks when needed:

```sh
make check
```

## Modules

Each module has its own README with detailed documentation on configuration, CLI usage, and pipeline steps.

| Module | Path | Description |
|---|---|---|
| **Data** | [`modules/data/`](modules/data/README.md) | Download, preprocess, cluster, score, assemble, and upload training datasets |
| **Pretraining** | [`modules/pretrain/`](modules/pretrain/README.md) | Protein language model pretraining |
| **Evaluation** | [`modules/evaluate/`](modules/evaluate/README.md) | Evaluate pretrained protein language models on downstream tasks |

## Scope

This repo is not a polished product package. It is a development sandbox for driving the monorepo/package split and validating migration steps.
