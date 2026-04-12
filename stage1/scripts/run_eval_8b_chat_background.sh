#!/usr/bin/env bash
set -euo pipefail

# Background wrapper for Phase 6: 8B chat eval + error shift rebuild
# Uses the shared run_train_eval_background.sh supervisor with watchdog + ntfy.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export RUN_SCRIPT="${RUN_SCRIPT:-${STAGE1_ROOT}/scripts/run_eval_8b_chat_and_errorshift.sh}"
export LOG_DIR="${LOG_DIR:-${STAGE1_ROOT}/logs}"
export RUN_LOG="${RUN_LOG:-${LOG_DIR}/eval_8b_chat_and_errorshift.log}"
export SUP_LOG="${SUP_LOG:-${LOG_DIR}/eval_8b_chat_and_errorshift_background.log}"
export EXIT_CODE_FILE="${EXIT_CODE_FILE:-${LOG_DIR}/eval_8b_chat_and_errorshift_exit_code.txt}"

export MAIN_PID_FILE="${MAIN_PID_FILE:-${LOG_DIR}/eval_8b_chat_main.pid}"
export WATCHDOG_PID_FILE="${WATCHDOG_PID_FILE:-${LOG_DIR}/eval_8b_chat_watchdog.pid}"
export NTFY_PID_FILE="${NTFY_PID_FILE:-${LOG_DIR}/eval_8b_chat_ntfy.pid}"

export NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
export NTFY_TOPIC="${NTFY_TOPIC:-finqa-stage1-8b-chat-eval}"
export ENABLE_NTFY="${ENABLE_NTFY:-true}"

mkdir -p "${LOG_DIR}"

exec bash "${SCRIPT_DIR}/run_train_eval_background.sh" "$@"
