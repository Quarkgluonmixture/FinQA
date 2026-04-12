#!/usr/bin/env bash
set -euo pipefail

# Background wrapper for Phase 7: 8B full-steps retrain + dual eval
# Watchdog + ntfy supervisor.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export RUN_SCRIPT="${RUN_SCRIPT:-${STAGE1_ROOT}/scripts/run_8b_fullsteps_train_eval.sh}"
export LOG_DIR="${LOG_DIR:-${STAGE1_ROOT}/logs}"
export RUN_LOG="${RUN_LOG:-${LOG_DIR}/8b_fullsteps_train_eval.log}"
export SUP_LOG="${SUP_LOG:-${LOG_DIR}/8b_fullsteps_background.log}"
export EXIT_CODE_FILE="${EXIT_CODE_FILE:-${LOG_DIR}/8b_fullsteps_exit_code.txt}"

export MAIN_PID_FILE="${MAIN_PID_FILE:-${LOG_DIR}/8b_fullsteps_main.pid}"
export WATCHDOG_PID_FILE="${WATCHDOG_PID_FILE:-${LOG_DIR}/8b_fullsteps_watchdog.pid}"
export NTFY_PID_FILE="${NTFY_PID_FILE:-${LOG_DIR}/8b_fullsteps_ntfy.pid}"

export NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
export NTFY_TOPIC="${NTFY_TOPIC:-finqa-stage1-8b-fullsteps}"
export ENABLE_NTFY="${ENABLE_NTFY:-true}"

mkdir -p "${LOG_DIR}"

exec bash "${SCRIPT_DIR}/run_train_eval_background.sh" "$@"
