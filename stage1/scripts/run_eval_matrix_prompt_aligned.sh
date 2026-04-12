#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_ROOT="${STAGE1_ROOT}/../finqa_baseline"

if [[ -x "${STAGE1_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${STAGE1_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "[error] no usable python found for stage1." >&2
  exit 127
fi

if [[ -x "${BASELINE_ROOT}/.venv/bin/python" ]]; then
  EVAL_PYTHON_BIN="${BASELINE_ROOT}/.venv/bin/python"
else
  EVAL_PYTHON_BIN="${PYTHON_BIN}"
fi

if [[ -f "${BASELINE_ROOT}/scripts/apply_dgx_spark_quirks.sh" ]]; then
  # shellcheck source=/dev/null
  source "${BASELINE_ROOT}/scripts/apply_dgx_spark_quirks.sh"
fi

HF_CACHE_ROOT="${HF_CACHE_ROOT:-/home/jiaming/workspace/.cache/huggingface}"
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

SEED="${SEED:-42}"
EVAL_SETTING="${EVAL_SETTING:-oracle}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:--1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_PROMPT_PROTOCOL="${EVAL_PROMPT_PROTOCOL:-stage1_train_text}"

MODEL_4B="${MODEL_4B:-Qwen/Qwen3-4B}"
MODEL_8B="${MODEL_8B:-Qwen/Qwen3-8B}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${STAGE1_ROOT}/outputs/sft_clean_strict}"
RESULTS_ROOT="${RESULTS_ROOT:-${BASELINE_ROOT}/results/sft_clean_strict_prompt_aligned_eval}"
MANIFEST_PATH="${MANIFEST_PATH:-${RESULTS_ROOT}/run_manifest.jsonl}"
SFT_MANIFEST_PATH="${SFT_MANIFEST_PATH:-${RESULTS_ROOT}/run_manifest_sft_only.jsonl}"
CLEAN_EXISTING_RESULTS="${CLEAN_EXISTING_RESULTS:-false}"
EVAL_RETRY_MAX="${EVAL_RETRY_MAX:-5}"
EVAL_RETRY_SLEEP="${EVAL_RETRY_SLEEP:-20}"

mkdir -p "${RESULTS_ROOT}"
rm -f "${MANIFEST_PATH}" "${SFT_MANIFEST_PATH}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

sanitize_model() {
  local model="$1"
  echo "${model}" | tr '/:' '__'
}

ensure_eval_dependencies() {
  log "Checking eval dependencies in ${EVAL_PYTHON_BIN} ..."
  local missing
  missing="$(${EVAL_PYTHON_BIN} - <<'PY'
import importlib.util
required = ["peft", "math_verify", "transformers", "accelerate"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
print(" ".join(missing))
PY
)"
  if [[ -n "${missing}" ]]; then
    log "Installing missing eval deps: ${missing}"
    "${EVAL_PYTHON_BIN}" -m pip install ${missing}
  else
    log "Eval dependencies are ready."
  fi
}

append_manifest() {
  local run_name="$1"
  local model_name="$2"
  local model_size="$3"
  local supervision_style="$4"
  local train_size="$5"
  local adapter_path="$6"
  local eval_jsonl="$7"
  local results_dir="$8"
  local is_sft="$9"

  "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${run_name}" "${model_name}" "${model_size}" "${supervision_style}" "${train_size}" "${adapter_path}" "${eval_jsonl}" "${results_dir}" "${EVAL_PROMPT_PROTOCOL}" "${is_sft}" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest,
    run_name,
    model_name,
    model_size,
    supervision_style,
    train_size,
    adapter_path,
    eval_jsonl,
    results_dir,
    prompt_protocol,
    is_sft,
) = sys.argv[1:]

payload = {
    "run_name": run_name,
    "phase": "prompt_aligned_eval",
    "model_name": model_name,
    "model_size": model_size,
    "supervision_style": supervision_style,
    "train_size": train_size,
    "adapter_path": adapter_path,
    "eval_jsonl": eval_jsonl,
    "results_dir": results_dir,
    "prompt_protocol": prompt_protocol,
    "is_sft": str(is_sft).lower() == "true",
    "data_policy": "strict_consistent_only",
}

path = Path(manifest)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
}

run_eval() {
  local run_name="$1"
  local model_name="$2"
  local model_size="$3"
  local supervision_style="$4"
  local train_size="$5"
  local adapter_path="$6"
  local is_sft="$7"

  local results_dir="${RESULTS_ROOT}/${run_name}"
  mkdir -p "${results_dir}"

  local model_safe
  model_safe="$(sanitize_model "${model_name}")"
  local eval_jsonl="${results_dir}/finqa_${model_safe}_${EVAL_SETTING}_${EVAL_SPLIT}.jsonl"

  if [[ "${CLEAN_EXISTING_RESULTS}" == "true" ]]; then
    rm -f "${eval_jsonl}" "${results_dir}/summary.json" "${results_dir}/error_cases.md"
  fi

  log "Eval run=${run_name} model=${model_name} adapter=${adapter_path:-<none>} style=${supervision_style}"
  local attempt=1
  local rc=0
  while true; do
    log "Eval attempt ${attempt}/${EVAL_RETRY_MAX} for run=${run_name}"
    set +e
    (
      cd "${BASELINE_ROOT}"
      args=(
        eval_finqa.py
        --model_name "${model_name}"
        --setting "${EVAL_SETTING}"
        --split "${EVAL_SPLIT}"
        --cache_dir "${HF_CACHE_ROOT}"
        --results_dir "${results_dir}"
        --evaluator "math_verify"
        --answer_format "final_answer_tag"
        --final_answer_tag "FINAL_ANSWER"
        --prompt_protocol "${EVAL_PROMPT_PROTOCOL}"
        --no-enable_thinking
        --max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
        --seed "${SEED}"
        --num_samples "${EVAL_NUM_SAMPLES}"
      )

      if [[ -n "${adapter_path}" ]]; then
        args+=(--adapter_path "${adapter_path}")
      fi

      "${EVAL_PYTHON_BIN}" "${args[@]}"
    )
    rc=$?
    set -e

    if [[ ${rc} -eq 0 ]]; then
      break
    fi

    if [[ ${attempt} -ge ${EVAL_RETRY_MAX} ]]; then
      log "[error] Eval run=${run_name} failed after ${attempt} attempts (exit=${rc})."
      return "${rc}"
    fi

    # 137/143 are common when process gets killed externally (OOM/timeout).
    log "[warn] Eval run=${run_name} failed with exit=${rc}; will retry after ${EVAL_RETRY_SLEEP}s. Existing jsonl will be resumed."
    sleep "${EVAL_RETRY_SLEEP}"
    attempt=$((attempt + 1))
  done

  append_manifest \
    "${run_name}" "${model_name}" "${model_size}" "${supervision_style}" "${train_size}" \
    "${adapter_path}" "${eval_jsonl}" "${results_dir}" "${is_sft}"
}

ensure_checkpoint_exists() {
  local run_name="$1"
  local ckpt_path="$2"
  if [[ ! -d "${ckpt_path}" ]]; then
    log "[error] missing checkpoint for ${run_name}: ${ckpt_path}"
    exit 1
  fi
}

run_error_shift_report() {
  log "Stage 3: mandatory error-shift report (SFT 6 runs only)"

  "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${SFT_MANIFEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

rows = []
with manifest_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("supervision_style") in {"answer_only", "formula_rationale"}:
            rows.append(row)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[saved] {out_path}")
PY

  "${PYTHON_BIN}" "${STAGE1_ROOT}/scripts/analyze_error_shift.py" \
    --manifest_jsonl "${SFT_MANIFEST_PATH}" \
    --output_md "${RESULTS_ROOT}/error_shift_report.md" \
    --output_json "${RESULTS_ROOT}/error_shift_report.json"
}

main() {
  log "Stage 0: dependency check"
  ensure_eval_dependencies
  log "Using prompt_protocol=${EVAL_PROMPT_PROTOCOL}"

  local ckpt_4b_answer="${CHECKPOINT_ROOT}/4b_clean_full_answer_only/checkpoint-last"
  local ckpt_4b_formula="${CHECKPOINT_ROOT}/4b_clean_full_formula_rationale/checkpoint-last"
  local ckpt_8b_answer="${CHECKPOINT_ROOT}/8b_clean_full_answer_only/checkpoint-last"
  local ckpt_8b_formula="${CHECKPOINT_ROOT}/8b_clean_full_formula_rationale/checkpoint-last"
  local ckpt_8b_250="${CHECKPOINT_ROOT}/8b_clean_250_answer_only/checkpoint-last"
  local ckpt_8b_1000="${CHECKPOINT_ROOT}/8b_clean_1000_answer_only/checkpoint-last"

  ensure_checkpoint_exists "4b_clean_full_answer_only" "${ckpt_4b_answer}"
  ensure_checkpoint_exists "4b_clean_full_formula_rationale" "${ckpt_4b_formula}"
  ensure_checkpoint_exists "8b_clean_full_answer_only" "${ckpt_8b_answer}"
  ensure_checkpoint_exists "8b_clean_full_formula_rationale" "${ckpt_8b_formula}"
  ensure_checkpoint_exists "8b_clean_250_answer_only" "${ckpt_8b_250}"
  ensure_checkpoint_exists "8b_clean_1000_answer_only" "${ckpt_8b_1000}"

  log "Stage 1: re-evaluate 6 SFT checkpoints"
  run_eval "4b_clean_full_answer_only" "${MODEL_4B}" "4B" "answer_only" "full" "${ckpt_4b_answer}" "true"
  run_eval "4b_clean_full_formula_rationale" "${MODEL_4B}" "4B" "formula_rationale" "full" "${ckpt_4b_formula}" "true"
  run_eval "8b_clean_full_answer_only" "${MODEL_8B}" "8B" "answer_only" "full" "${ckpt_8b_answer}" "true"
  run_eval "8b_clean_full_formula_rationale" "${MODEL_8B}" "8B" "formula_rationale" "full" "${ckpt_8b_formula}" "true"
  run_eval "8b_clean_250_answer_only" "${MODEL_8B}" "8B" "answer_only" "250" "${ckpt_8b_250}" "true"
  run_eval "8b_clean_1000_answer_only" "${MODEL_8B}" "8B" "answer_only" "1000" "${ckpt_8b_1000}" "true"

  log "Stage 2: prompt-aligned zero-shot baselines"
  run_eval "4b_zeroshot_prompt_aligned" "${MODEL_4B}" "4B" "zero_shot" "0" "" "false"
  run_eval "8b_zeroshot_prompt_aligned" "${MODEL_8B}" "8B" "zero_shot" "0" "" "false"

  run_error_shift_report

  log "All done."
  log "Manifest (all runs): ${MANIFEST_PATH}"
  log "Manifest (SFT only): ${SFT_MANIFEST_PATH}"
  log "Error-shift markdown: ${RESULTS_ROOT}/error_shift_report.md"
}

main "$@"
