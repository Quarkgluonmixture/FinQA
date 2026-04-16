# finqa_baseline

Evaluation pipeline for FinQA zero-shot and post-training adapter runs.

## Setup

```bash
bash setup.sh
source .venv/bin/activate
```

## Cache Configuration

Defaults are portable and environment-based.

```bash
export CACHE_DIR="${CACHE_DIR:-$HOME/.cache/huggingface}"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$CACHE_DIR/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_DIR/transformers}"
```

## Single Evaluation Example

```bash
python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --split test \
  --setting oracle \
  --cache_dir "${CACHE_DIR:-$HOME/.cache/huggingface}"
```

## Adapter Evaluation Example

```bash
python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --adapter_path /path/to/checkpoint-last \
  --setting oracle \
  --split test \
  --no-enable_thinking \
  --answer_format final_answer_tag \
  --final_answer_tag FINAL_ANSWER
```

## Matrix Entrypoints

- `run_verification_matrix.sh`: baseline matrix and robust verification report
- `run_verification_matrix_loop.sh`: continuous loop wrapper
- `run_forever_no_guard.sh`: backward-compatible wrapper (deprecated)

## Output Policy

Runtime outputs are generated under `results/` and `logs/` but are excluded from version control except for directory keep files.

## Summary Fields

`summary.json` includes compatibility metrics such as:

- `accuracy_mathverify`
- `accuracy_legacy`
- `accuracy_legacy_base`
- `parse_fail_rate_mathverify`
- `parse_fail_rate_legacy`
- `tag_status_counts`
