#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export RUN_SCRIPT="${RUN_SCRIPT:-${STAGE1_ROOT}/scripts/run_eval_matrix_prompt_aligned.sh}"
export LOG_DIR="${LOG_DIR:-${STAGE1_ROOT}/logs}"
export RUN_LOG="${RUN_LOG:-${LOG_DIR}/eval_prompt_aligned.log}"
export SUP_LOG="${SUP_LOG:-${LOG_DIR}/eval_prompt_aligned_background.log}"
export EXIT_CODE_FILE="${EXIT_CODE_FILE:-${LOG_DIR}/eval_prompt_aligned_exit_code.txt}"

export MAIN_PID_FILE="${MAIN_PID_FILE:-${LOG_DIR}/eval_prompt_aligned_main.pid}"
export WATCHDOG_PID_FILE="${WATCHDOG_PID_FILE:-${LOG_DIR}/eval_prompt_aligned_watchdog.pid}"
export NTFY_PID_FILE="${NTFY_PID_FILE:-${LOG_DIR}/eval_prompt_aligned_ntfy.pid}"

export NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
export NTFY_TOPIC="${NTFY_TOPIC:-finqa-stage1-evalfix}"
export ENABLE_NTFY="${ENABLE_NTFY:-true}"

mkdir -p "${LOG_DIR}"

exec bash "${SCRIPT_DIR}/run_train_eval_background.sh" "$@"
