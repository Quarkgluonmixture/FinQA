#!/usr/bin/env bash
set -euo pipefail

# Phase 7: Retrain 8B with max_steps=-1 (full 2 epochs) then evaluate with both protocols
#
# 1. Train 8b_clean_full_answer_only_fullsteps (~1638 steps, ~3h)
# 2. Eval with chat_default protocol (~1.5h)
# 3. Eval with stage1_train_text protocol (~1.5h)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_ROOT="${STAGE1_ROOT}/../finqa_baseline"

if [[ -x "${BASELINE_ROOT}/.venv/bin/python" ]]; then
  EVAL_PYTHON_BIN="${BASELINE_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  EVAL_PYTHON_BIN="$(command -v python3)"
else
  echo "[error] no usable python found." >&2; exit 127
fi

if [[ -f "${BASELINE_ROOT}/scripts/apply_dgx_spark_quirks.sh" ]]; then
  source "${BASELINE_ROOT}/scripts/apply_dgx_spark_quirks.sh"
fi

HF_CACHE_ROOT="${HF_CACHE_ROOT:-${HOME}/.cache/huggingface}"
HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"
export HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE

HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.config/huggingface/token}"
if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_FILE}" ]]; then
  HF_TOKEN="$(cat "${HF_TOKEN_FILE}" | tr -d '[:space:]')"
  export HF_TOKEN
fi

SEED=42
RUN_NAME="8b_clean_full_answer_only_fullsteps"
CONFIG="${STAGE1_ROOT}/configs/generated_clean_strict/${RUN_NAME}.yaml"
CHECKPOINT_DIR="${STAGE1_ROOT}/outputs/sft_clean_strict/${RUN_NAME}/checkpoint-last"
RESULTS_ROOT="${BASELINE_ROOT}/results/sft_clean_strict_fullsteps_eval"
MODEL_8B="Qwen/Qwen3-8B"

EVAL_RETRY_MAX=5
EVAL_RETRY_SLEEP=30

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

summary_metric() {
  local summary_json="$1"
  local metric_key="$2"
  "${EVAL_PYTHON_BIN}" - "${STAGE1_ROOT}" "${summary_json}" "${metric_key}" <<'PY'
import sys
from pathlib import Path

stage1_root = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
metric_key = sys.argv[3]
sys.path.insert(0, str(stage1_root / "scripts"))

from summary_utils import latest_run_record

record = latest_run_record(summary_json)
print(record.get(metric_key))
PY
}

mkdir -p "${RESULTS_ROOT}"

# ============================================================
# Step 1: Train
# ============================================================
log "=========================================="
log "Phase 7: 8B full-steps retrain + dual eval"
log "=========================================="

if [[ -d "${CHECKPOINT_DIR}" ]]; then
  log "[skip] Checkpoint already exists: ${CHECKPOINT_DIR}"
  log "Skipping training, proceeding to eval."
else
  log "Step 1: Training ${RUN_NAME} (max_steps=-1, 2 epochs, ~1638 steps)"
  (
    cd "${STAGE1_ROOT}"
    bash scripts/run_train.sh "${CONFIG}"
  )
  log "Training complete."

  if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    # Try to find the actual last checkpoint
    LAST_CKPT=$(ls -d "${STAGE1_ROOT}/outputs/sft_clean_strict/${RUN_NAME}/checkpoint-"* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [[ -n "${LAST_CKPT}" ]]; then
      ln -sfn "${LAST_CKPT}" "${CHECKPOINT_DIR}"
      log "Linked checkpoint-last -> $(basename "${LAST_CKPT}")"
    else
      log "[error] No checkpoint found after training!"
      exit 1
    fi
  fi
fi

# ============================================================
# Step 2: Eval with chat_default
# ============================================================
run_eval() {
  local eval_name="$1" prompt_protocol="$2" results_dir="$3"

  mkdir -p "${results_dir}"

  if [[ -f "${results_dir}/summary.json" ]]; then
    log "[skip] ${eval_name} already has summary.json"
    return 0
  fi

  log "Eval: ${eval_name} protocol=${prompt_protocol}"

  local attempt=1 rc=0
  while true; do
    log "  Attempt ${attempt}/${EVAL_RETRY_MAX}"
    set +e
    (
      cd "${BASELINE_ROOT}"
      "${EVAL_PYTHON_BIN}" eval_finqa.py \
        --model_name "${MODEL_8B}" \
        --adapter_path "${CHECKPOINT_DIR}" \
        --setting oracle \
        --split test \
        --cache_dir "${HF_CACHE_ROOT}" \
        --results_dir "${results_dir}" \
        --evaluator math_verify \
        --answer_format final_answer_tag \
        --final_answer_tag FINAL_ANSWER \
        --prompt_protocol "${prompt_protocol}" \
        --no-enable_thinking \
        --max_new_tokens 256 \
        --seed "${SEED}" \
        --num_samples -1
    )
    rc=$?
    set -e

    if [[ ${rc} -eq 0 ]]; then break; fi
    if [[ ${attempt} -ge ${EVAL_RETRY_MAX} ]]; then
      log "[error] ${eval_name} failed after ${attempt} attempts."
      return "${rc}"
    fi
    log "[warn] ${eval_name} failed (exit=${rc}); retry after ${EVAL_RETRY_SLEEP}s."
    sleep "${EVAL_RETRY_SLEEP}"
    attempt=$((attempt + 1))
  done
}

log "Step 2: Eval with chat_default protocol"
run_eval "${RUN_NAME}_chat" "chat_default" "${RESULTS_ROOT}/${RUN_NAME}_chat"

log "Step 3: Eval with stage1_train_text protocol"
run_eval "${RUN_NAME}_text" "stage1_train_text" "${RESULTS_ROOT}/${RUN_NAME}_text"

# ============================================================
# Summary
# ============================================================
log "=========================================="
log "All done. Results:"
log "  Chat eval: ${RESULTS_ROOT}/${RUN_NAME}_chat/summary.json"
log "  Text eval: ${RESULTS_ROOT}/${RUN_NAME}_text/summary.json"
log "=========================================="

# Print key metrics
for protocol in chat text; do
  sfile="${RESULTS_ROOT}/${RUN_NAME}_${protocol}/summary.json"
  if [[ -f "${sfile}" ]]; then
    log "--- ${protocol} ---"
    acc_base="$(summary_metric "${sfile}" "accuracy_base")"
    acc_adjusted="$(summary_metric "${sfile}" "accuracy_adjusted")"
    parse_fail="$(summary_metric "${sfile}" "parse_fail_rate")"
    python3 -c "
acc_base = float('${acc_base}')
acc_adjusted = float('${acc_adjusted}')
parse_fail = float('${parse_fail}')
print(f'  acc_base:     {acc_base:.4f}')
print(f'  acc_adjusted: {acc_adjusted:.4f}')
print(f'  parse_fail:   {parse_fail:.4f}')
"
  fi
done
