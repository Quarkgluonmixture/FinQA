# Reproducibility Guide

This document describes the minimal, reproducible workflow for this repository.

## 1. Environment

Use Python 3.10+ (3.12 tested).

### Training environment (`stage1`)

```bash
cd stage1
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

### Evaluation environment (`finqa_baseline`)

```bash
cd finqa_baseline
bash setup.sh
```

## 2. Cache and Authentication

All scripts default to `${HOME}/.cache/huggingface`.

Optional overrides:

```bash
export HF_CACHE_ROOT="$HOME/.cache/huggingface"
export HF_HOME="$HF_CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT/transformers"
export HF_TOKEN_FILE="$HOME/.config/huggingface/token"
```

## 3. Minimal Sanity Reproduction

```bash
cd stage1
bash scripts/run_debug.sh
```

This checks config load, preprocessing, training, checkpoint write, and dry-run post-train inference.

## 4. Main Experiment Entrypoints

### Baseline evaluation matrix

```bash
cd finqa_baseline
bash run_verification_matrix.sh
```

### Stage-1 clean strict training/eval matrix

```bash
cd stage1
bash scripts/run_train_eval_matrix_clean_strict.sh
```

### Prompt-aligned evaluation matrix

```bash
cd stage1
bash scripts/run_eval_matrix_prompt_aligned.sh
```

### 8B full-steps retrain and dual evaluation

```bash
cd stage1
bash scripts/run_8b_fullsteps_train_eval.sh
```

## 5. Summary Contract

Evaluation scripts must read metrics from the latest run record in `summary.json`:

- if `summary.json` has `runs`, consume `runs[-1]`
- otherwise consume top-level fields

The helper `stage1/scripts/summary_utils.py` is the canonical reader.

## 6. Result Snapshot

See `results_snapshot.md` for key numbers used in the report.
