#!/usr/bin/env bash
set -euo pipefail

# Wait until GPU is sufficiently free, then start FinQA verification matrix.
#
# Usage:
#   cd /home/jiaming/workspace/FinQA/finqa_baseline
#   nohup bash gpu_guard_autorun.sh > logs/gpu_guard_console.log 2>&1 &
#   echo $! > logs/gpu_guard_autorun.pid
#
# Optional env vars:
#   GPU_INDEX=0
#   CHECK_INTERVAL_SEC=30
#   MIN_FREE_RATIO=0.70
#   MIN_FREE_MIB=0
#   MAX_OTHER_PROCS=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
source "${ROOT_DIR}/scripts/apply_dgx_spark_quirks.sh"

GPU_INDEX="${GPU_INDEX:-0}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-30}"
MIN_FREE_RATIO="${MIN_FREE_RATIO:-0.70}"
MIN_FREE_MIB="${MIN_FREE_MIB:-0}"
MAX_OTHER_PROCS="${MAX_OTHER_PROCS:-0}"

LOG_DIR="${LOG_DIR:-logs}"
GUARD_LOG="${GUARD_LOG:-${LOG_DIR}/gpu_guard_autorun.log}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/run_verification_matrix_full.log}"
RUN_PID_FILE="${RUN_PID_FILE:-${LOG_DIR}/run_verification_matrix_full.pid}"

mkdir -p "${LOG_DIR}"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "${GUARD_LOG}"; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "ERROR: nvidia-smi not found."
  exit 1
fi

if pgrep -af "run_verification_matrix.sh|python eval_finqa.py" >/dev/null 2>&1; then
  log "Existing FinQA run detected; skip duplicate start."
  pgrep -af "run_verification_matrix.sh|python eval_finqa.py" | tee -a "${GUARD_LOG}"
  exit 0
fi

log "GPU guard started (gpu=${GPU_INDEX}, interval=${CHECK_INTERVAL_SEC}s, min_free_ratio=${MIN_FREE_RATIO}, min_free_mib=${MIN_FREE_MIB}, max_other_procs=${MAX_OTHER_PROCS})"

while true; do
  gpu_query="$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
  gpu_line="$(awk -F',' -v idx="${GPU_INDEX}" '$1+0==idx {gsub(/ /,"",$2); gsub(/ /,"",$3); print $2 " " $3}' <<<"${gpu_query}")"
  if [[ -z "${gpu_line}" ]]; then
    log "GPU index ${GPU_INDEX} not found; retry in ${CHECK_INTERVAL_SEC}s."
    sleep "${CHECK_INTERVAL_SEC}"
    continue
  fi

  total_mib="$(awk '{print $1}' <<<"${gpu_line}")"
  used_mib="$(awk '{print $2}' <<<"${gpu_line}")"
  memory_ok=0
  free_mib=0
  need_mib=0
  if [[ "${total_mib}" =~ ^[0-9]+$ && "${used_mib}" =~ ^[0-9]+$ ]]; then
    free_mib="$(( total_mib - used_mib ))"
    ratio_need_mib="$(awk -v t="${total_mib}" -v r="${MIN_FREE_RATIO}" 'BEGIN{printf "%.0f", t*r}')"
    need_mib="${ratio_need_mib}"
    if [[ "${MIN_FREE_MIB}" -gt "${need_mib}" ]]; then
      need_mib="${MIN_FREE_MIB}"
    fi
    if [[ "${free_mib}" -ge "${need_mib}" ]]; then
      memory_ok=1
    fi
  else
    # Some hosts (e.g. GB10 with "Not Supported" memory telemetry) return N/A.
    # In this case we fall back to process-count gating.
    memory_ok=1
  fi

  proc_query="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  other_proc_count="$(awk 'NF{print $1}' <<<"${proc_query}" | sort -u | wc -l | tr -d ' ')"
  [[ -z "${other_proc_count}" ]] && other_proc_count="0"

  log "GPU${GPU_INDEX}: free=${free_mib}MiB / total=${total_mib}MiB, required_free>=${need_mib}MiB, compute_procs=${other_proc_count} (limit<=${MAX_OTHER_PROCS})"

  if [[ "${memory_ok}" -eq 1 && "${other_proc_count}" -le "${MAX_OTHER_PROCS}" ]]; then
    break
  fi

  sleep "${CHECK_INTERVAL_SEC}"
done

log "GPU condition satisfied; launching run_verification_matrix.sh"

source .venv/bin/activate
nohup ./run_verification_matrix.sh \
  > "${RUN_LOG}" 2>&1 &

run_pid="$!"
echo "${run_pid}" > "${RUN_PID_FILE}"
log "Launched PID=${run_pid}; run_log=${RUN_LOG}; pid_file=${RUN_PID_FILE}"
