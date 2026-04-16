# Stage 1 SFT Pipeline

This directory contains the training pipeline used for task-specific SFT experiments.

## Contents

- `train_sft.py`: main training entrypoint
- `run_infer.py`: post-training dry-run inference entrypoint
- `configs/`: base configs (`debug.yaml`, `small.yaml`, `full.yaml`)
- `scripts/`: orchestration scripts and utilities
- `src/`: preprocessing, data loading, config, and trainer modules
- `data/`: minimal tracked reproducibility data (debug samples, subset IDs, summaries)

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

## Quick Sanity Check

```bash
bash scripts/run_debug.sh
```

Expected outputs:

- `outputs/debug_run/checkpoint-last`
- `outputs/debug_run/infer_predictions.jsonl`
- `outputs/debug_run/infer_summary.json`

## Core Scripts

- `scripts/run_train.sh`: run training with a single config
- `scripts/run_train_eval_matrix.sh`: matrix orchestration
- `scripts/run_train_eval_matrix_clean_strict.sh`: strict-clean matrix
- `scripts/run_eval_matrix_prompt_aligned.sh`: prompt-aligned evaluation matrix
- `scripts/run_8b_fullsteps_train_eval.sh`: full-steps retrain + dual evaluation

## Inference Mode Compatibility

The canonical dry-run mode is:

- `dry_run_echo_reference`

Legacy value `smoke_echo_gold` is still accepted and mapped automatically with a deprecation warning.

## Config Contract

- Paths in configs should be repository-relative.
- Cache and machine-specific paths must be injected via environment variables.
- Generated configs are runtime artifacts and are not tracked.

## Notes

Large runtime artifacts are intentionally excluded from version control:

- `outputs/`
- `logs/`
- `reports/`
- generated configs and regenerated large data snapshots
