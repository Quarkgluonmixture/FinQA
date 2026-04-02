# COMP0087 Stage 1

## Overview

This repository contains the Stage 1 SFT pipeline for the COMP0087 group project.

The current integrated version supports:

- config-based training entry
- debug / small / full config files
- data loading and preprocessing
- LoRA-based Stage 1 training
- checkpoint saving
- minimal post-training inference verification

The debug smoke pipeline has been verified locally.

## Repository Structure

- `configs/`: yaml configs for debug / small / full runs
- `data/`: debug data and dataset preparation scripts
- `outputs/`: local checkpoints and run outputs
- `scripts/`: helper scripts for preprocessing / debug / training
- `src/`: core implementation
- `make_ablation_configs.py`: helper script for config generation
- `member6_log_and_run_guide.md`: additional run notes
- `requirements.txt`: python dependencies
- `run_infer.py`: minimal post-training inference entry
- `train_sft.py`: main Stage 1 training entry

## Environment Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

If your machine does not provide `python` (for example DGX Spark), use `python3`
or create `.venv` and run via `./.venv/bin/python`.

Recommended setup on DGX Spark:

```bash
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

## Config Files

Current config files include:

- `configs/debug.yaml` — smallest local smoke test config
- `configs/small.yaml` — small-scale run config
- `configs/full.yaml` — full run config

For first-time verification, always start with:

```text
configs/debug.yaml
```

## Data

### Debug data

The debug smoke test uses:

```text
data/debug/train.jsonl
```

### Data preparation scripts

The repository also includes data preparation utilities under `data/`, such as:

- `prepare_finqa.py`
- `prepare_convfinqa.py`
- `prepare_multihiertt.py`
- `merge_datasets.py`

## Main Entry Points

### Training entry

```bash
bash scripts/run_debug.sh
```

### Inference entry

```bash
python3 run_infer.py --help
```

## Verified Debug Smoke Pipeline

The following command has been verified locally:

```bash
bash scripts/run_debug.sh
```

This debug run successfully completed:

- config loading
- debug data loading
- preprocessing
- trainer initialization
- training
- checkpoint saving
- automatic minimal post-training inference check

## Manual Minimal Inference Check

The following command has also been verified locally:

```bash
python3 run_infer.py \
  --config configs/debug.yaml \
  --checkpoint_dir outputs/debug_run/checkpoint-last \
  --input_file data/debug/train.jsonl \
  --output_file outputs/debug_run/manual_infer_predictions.jsonl
```

This command successfully generated prediction output and summary files.

## Expected Outputs

For a verified debug run, outputs are written under:

```text
outputs/debug_run/
```

Typical files include:

- `checkpoint-5/`
- `checkpoint-last/`
- `infer_predictions.jsonl`
- `manual_infer_predictions.jsonl`
- `infer_summary.json`
- `run_meta.json`
- `trainer_state.json`
- `processed_preview.jsonl`

These files are local run artifacts and should generally not be committed to GitHub.

## Example Workflow

### Step 1: install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: run debug training

```bash
bash scripts/run_debug.sh
```

### Step 3: run manual inference check

```bash
python3 run_infer.py \
  --config configs/debug.yaml \
  --checkpoint_dir outputs/debug_run/checkpoint-last \
  --input_file data/debug/train.jsonl \
  --output_file outputs/debug_run/manual_infer_predictions.jsonl
```

## Scripts

Helper scripts are located in `scripts/`.

Examples include:

- `scripts/run_debug.sh`
- `scripts/run_train.sh`
- `scripts/preprocess.py`
- `scripts/prompting.py`

## Notes for Team Members

- Use `configs/debug.yaml` first before trying larger configs.
- Do not upload large local checkpoints or output artifacts to GitHub.
- Keep `outputs/` ignored by git.
- Update README if entry points, config paths, or output paths change.
- Use the verified commands above as the current baseline smoke test.

## Current Status

Current integrated status:

- Stage 1 repository structure completed
- debug / small / full config files available
- training entry verified
- minimal inference entry verified
- debug smoke pipeline verified locally

## Recommended .gitignore

```gitignore
outputs/
*.pt
*.bin
*.safetensors
__pycache__/
*.pyc
.DS_Store
```

## Handover Note

This repository is currently in an integrated Stage 1 state with a verified local debug pipeline.

For handover or further testing, start from:

```bash
bash scripts/run_debug.sh
```
