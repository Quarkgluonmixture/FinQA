# FinQA Reproducibility Repository

This repository contains the code and minimal reproducibility assets for our COMP0087 FinQA project.

## Repository Layout

- `docs/paper/`: final report artifacts
- `docs/repro/`: reproducibility instructions and result snapshots
- `finqa_baseline/`: zero-shot and post-training evaluation pipeline
- `stage1/`: SFT training pipeline, config generation, and experiment orchestration

## Quick Start

### 1) Set up environments

```bash
cd finqa_baseline
bash setup.sh

cd ../stage1
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

### 2) Run a minimal end-to-end sanity check

```bash
cd stage1
bash scripts/run_debug.sh
```

Expected outputs are under `stage1/outputs/debug_run/`.

### 3) Run baseline FinQA evaluation

```bash
cd finqa_baseline
.venv/bin/python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --setting oracle \
  --split test \
  --cache_dir "${HF_HOME:-$HOME/.cache/huggingface}"
```

## Main Reproduction Entrypoints

- Baseline matrix: `finqa_baseline/run_verification_matrix.sh`
- Continuous baseline loop (new name): `finqa_baseline/run_verification_matrix_loop.sh`
- Stage-1 clean strict matrix: `stage1/scripts/run_train_eval_matrix_clean_strict.sh`
- Stage-1 prompt-aligned eval matrix: `stage1/scripts/run_eval_matrix_prompt_aligned.sh`
- Stage-1 8B full-steps retrain + eval: `stage1/scripts/run_8b_fullsteps_train_eval.sh`

## Path and Environment Contract

- No script requires machine-specific absolute paths.
- Cache roots default to `${HOME}/.cache/huggingface`.
- Override with environment variables when needed:
  - `HF_CACHE_ROOT`
  - `HF_HOME`
  - `HUGGINGFACE_HUB_CACHE`
  - `TRANSFORMERS_CACHE`

## Data and Artifact Policy

This repository tracks only minimal reproducibility files:

- small debug samples
- subset ID lists
- cleaning summaries
- source code, configs, and documentation

Large regenerated artifacts (logs, checkpoints, reports, generated configs, full intermediate datasets) are intentionally excluded from version control.

## Report

- Final report PDF: `docs/paper/0087_report-3.pdf`
- Supporting reproducibility notes: `docs/repro/README.md`
