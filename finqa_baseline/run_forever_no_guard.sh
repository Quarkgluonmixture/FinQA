#!/usr/bin/env bash
set -euo pipefail

# Aggressive mode: continuously run FinQA verification matrix without GPU guard.
# Suitable for shared machines when you explicitly choose to compete for resources.
#
# Start:
#   nohup bash run_forever_no_guard.sh > logs/run_forever_no_guard.console.log 2>&1 &
#   echo $! > logs/run_forever_no_guard.pid
#
# Stop:
#   kill "$(cat logs/run_forever_no_guard.pid)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${LOG_DIR:-logs}"
LOOP_LOG="${LOOP_LOG:-${LOG_DIR}/run_forever_no_guard.log}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/run_verification_matrix_full.log}"
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-15}"

mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "${LOOP_LOG}"; }

if [[ ! -x ".venv/bin/python" ]]; then
  log "ERROR: .venv/bin/python not found. Run: bash setup.sh"
  exit 127
fi

source .venv/bin/activate

export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export HF_HUB_DISABLE_XET=1

attempt=0
while true; do
  attempt=$((attempt + 1))
  log "Attempt ${attempt}: starting run_verification_matrix.sh"

  if bash run_verification_matrix.sh >> "${RUN_LOG}" 2>&1; then
    log "Attempt ${attempt}: completed successfully."
  else
    code=$?
    log "Attempt ${attempt}: exited with code=${code}."
  fi

  log "Sleeping ${RESTART_DELAY_SEC}s before next attempt."
  sleep "${RESTART_DELAY_SEC}"
done
