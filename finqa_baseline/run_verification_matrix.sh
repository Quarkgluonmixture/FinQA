#!/usr/bin/env bash
set -euo pipefail

# Re-run FinQA baseline with:
# - thinking=true (primary)
# - thinking=false (ablation)
# - FINAL_ANSWER tag format
# - math_verify evaluator
#
# Outputs are isolated per thinking mode to avoid file overwrite:
#   results/thinking_true
#   results/thinking_false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[error] no usable python found (.venv/bin/python, python3, python)." >&2
    exit 127
  fi
fi

CACHE_DIR="${CACHE_DIR:-/home/jiaming/workspace/.cache/huggingface}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
LOG_DIR="${LOG_DIR:-logs}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"

EVALUATOR="math_verify"
ANSWER_FORMAT="final_answer_tag"
FINAL_ANSWER_TAG="FINAL_ANSWER"

MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-4B")
SETTINGS=("oracle" "full")
THINKING_MODES=("true" "false")

mkdir -p "${CACHE_DIR}/hub" "${CACHE_DIR}/transformers"
mkdir -p "${RESULTS_ROOT}" "${LOG_DIR}"

export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
# DGX Spark machine quirks: these reduce CUDA init stalls/hangs on this host.
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export PYTORCH_NVML_BASED_CUDA_CHECK=1
# Avoid xet CAS range errors (HTTP 416) observed on this DGX host for large model shards.
export HF_HUB_DISABLE_XET=1

sanitize_model_name() {
  local model="$1"
  echo "${model}" | tr '/:' '__'
}

run_single_eval() {
  local thinking="$1"
  local model="$2"
  local setting="$3"

  local thinking_label="thinking_${thinking}"
  local results_dir="${RESULTS_ROOT}/${thinking_label}"
  local model_safe
  model_safe="$(sanitize_model_name "${model}")"
  local log_file="${LOG_DIR}/${thinking_label}_${model_safe}_${setting}_${SPLIT}.log"

  mkdir -p "${results_dir}"

  local thinking_flag="--enable_thinking"
  if [[ "${thinking}" == "false" ]]; then
    thinking_flag="--no-enable_thinking"
  fi

  echo "========================================"
  echo "Run: thinking=${thinking}, model=${model}, setting=${setting}, split=${SPLIT}"
  echo "Results dir: ${results_dir}"
  echo "Log: ${log_file}"
  echo "========================================"

  # shellcheck disable=SC2086
  "${PYTHON_BIN}" eval_finqa.py \
    --model_name "${model}" \
    --split "${SPLIT}" \
    --setting "${setting}" \
    --cache_dir "${CACHE_DIR}" \
    --results_dir "${results_dir}" \
    --seed "${SEED}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --num_samples "${NUM_SAMPLES}" \
    --evaluator "${EVALUATOR}" \
    --answer_format "${ANSWER_FORMAT}" \
    --final_answer_tag "${FINAL_ANSWER_TAG}" \
    ${thinking_flag} \
    2>&1 | tee "${log_file}"
}

run_regression_sanity() {
  local sanity_dir="${RESULTS_ROOT}/sanity"
  mkdir -p "${sanity_dir}"
  echo "Running regression sanity checks..."
  "${PYTHON_BIN}" regression_final_answer_mathverify.py \
    --results_dir "${sanity_dir}" \
    --output "${sanity_dir}/regression_final_answer_mathverify.md" \
    2>&1 | tee "${LOG_DIR}/regression_final_answer_mathverify.log"
}

generate_per_mode_reports() {
  for thinking in "${THINKING_MODES[@]}"; do
    local results_dir="${RESULTS_ROOT}/thinking_${thinking}"
    if [[ -f "${results_dir}/summary.json" ]]; then
      echo "Generating report for ${results_dir}..."
      "${PYTHON_BIN}" generate_report.py \
        --results_dir "${results_dir}" \
        --output "${results_dir}/final_report.md" \
        2>&1 | tee "${LOG_DIR}/report_thinking_${thinking}.log"
    else
      echo "Skip report for ${results_dir}: summary.json not found."
    fi
  done
}

main() {
  run_regression_sanity

  for thinking in "${THINKING_MODES[@]}"; do
    for model in "${MODELS[@]}"; do
      for setting in "${SETTINGS[@]}"; do
        run_single_eval "${thinking}" "${model}" "${setting}"
      done
    done
  done

  generate_per_mode_reports

  "${PYTHON_BIN}" build_robust_verification_report.py \
    --results_root "${RESULTS_ROOT}" \
    --output_md "${RESULTS_ROOT}/robust_verification_report.md" \
    --output_json "${RESULTS_ROOT}/robust_verification_summary.json" \
    2>&1 | tee "${LOG_DIR}/robust_verification_report.log"

  echo ""
  echo "Done."
  echo "Main report: ${RESULTS_ROOT}/robust_verification_report.md"
  echo "Summary json: ${RESULTS_ROOT}/robust_verification_summary.json"
}

main "$@"
