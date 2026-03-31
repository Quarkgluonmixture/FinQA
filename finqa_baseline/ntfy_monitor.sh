#!/usr/bin/env bash
# ntfy_monitor.sh - Monitor FinQA baseline progress and send push notifications via ntfy.sh
#
# Usage:
#   bash ntfy_monitor.sh                          # watch default log with default topic
#   NTFY_TOPIC=my-topic bash ntfy_monitor.sh      # custom topic
#   NOTIFY_EVERY=100 bash ntfy_monitor.sh         # notify every 100 samples
#   WATCHDOG_TIMEOUT=1800 bash ntfy_monitor.sh    # alert if no progress for 30 min (default)
#
# Run in background:
#   nohup bash ntfy_monitor.sh > logs/ntfy_monitor.log 2>&1 &
#   echo $! > logs/ntfy_monitor.pid
#
# Stop:
#   kill "$(cat logs/ntfy_monitor.pid)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NTFY_TOPIC="${NTFY_TOPIC:-finqa-baseline}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/run_verification_matrix_full.log}"
NOTIFY_EVERY="${NOTIFY_EVERY:-50}"      # send notification every N samples
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1800}"  # alert if no tqdm line for this many seconds (default 30min)
WATCHDOG_CHECK_INTERVAL=60              # check every 60 seconds

last_notified=0
current_run=""

# Temp file shared between main loop and watchdog subprocess
LAST_PROGRESS_FILE="/tmp/finqa_ntfy_last_progress_$$"
echo "$(date +%s)" > "${LAST_PROGRESS_FILE}"

ts() { date +"%H:%M:%S"; }
log() { echo "[$(ts)] [ntfy_monitor] $*"; }

send_ntfy() {
    local title="$1"
    local msg="$2"
    local priority="${3:-default}"
    curl -s -o /dev/null \
        -H "Title: ${title}" \
        -H "Priority: ${priority}" \
        -d "${msg}" \
        "${NTFY_SERVER}/${NTFY_TOPIC}"
}

if ! command -v curl &>/dev/null; then
    echo "[ntfy_monitor] ERROR: curl not found." >&2
    exit 1
fi

log "Watching  : ${LOG_FILE}"
log "Topic     : ${NTFY_SERVER}/${NTFY_TOPIC}"
log "Interval  : every ${NOTIFY_EVERY} samples"
log "Watchdog  : alert if no progress for ${WATCHDOG_TIMEOUT}s"

# ── Watchdog subprocess ───────────────────────────────────────────────────────
# Runs in background; reads last-progress timestamp from temp file.
# Sends urgent alert if no tqdm line seen for WATCHDOG_TIMEOUT seconds.
(
    last_alert_ts=0
    while true; do
        sleep "${WATCHDOG_CHECK_INTERVAL}"
        [[ -f "${LAST_PROGRESS_FILE}" ]] || continue
        last_ts=$(cat "${LAST_PROGRESS_FILE}" 2>/dev/null) || continue
        now=$(date +%s)
        elapsed=$(( now - last_ts ))
        # Alert once per WATCHDOG_TIMEOUT period (avoid repeated spam)
        if (( elapsed >= WATCHDOG_TIMEOUT && now - last_alert_ts >= WATCHDOG_TIMEOUT )); then
            mins=$(( elapsed / 60 ))
            send_ntfy "FinQA Watchdog" "No progress for ${mins}min — process may be stuck!" "urgent"
            echo "[$(date +%H:%M:%S)] [ntfy_monitor] WATCHDOG: no progress for ${mins}min" >&2
            last_alert_ts=$now
        fi
    done
) &
WATCHDOG_PID=$!
trap "kill ${WATCHDOG_PID} 2>/dev/null; rm -f ${LAST_PROGRESS_FILE}" EXIT

# ── Startup: recover context from existing log ────────────────────────────────
if [[ -f "${LOG_FILE}" ]]; then
    run_line=$(grep -o 'Run: thinking=[^,]*, model=[^,]*, setting=[^,]*, split=.*' "${LOG_FILE}" | tail -1 || true)
    if [[ "$run_line" =~ Run:\ thinking=([^,]+),\ model=([^,]+),\ setting=([^,]+),\ split=(.+) ]]; then
        short_model="${BASH_REMATCH[2]##*/}"
        current_run="${short_model} ${BASH_REMATCH[3]} think=${BASH_REMATCH[1]}"
        log "Recovered run context: ${current_run}"
    fi
    last_line=$(tr '\r' '\n' < "${LOG_FILE}" | grep -o 'Evaluating FinQA:.*| *[0-9]*/[0-9]* \[' | tail -1 || true)
    if [[ "$last_line" =~ \|\ *([0-9]+)/([0-9]+)\ \[ ]]; then
        last_notified="${BASH_REMATCH[1]}"
        log "Current position: ${last_notified}/${BASH_REMATCH[2]}"
        # Reset watchdog clock to now so we don't immediately false-alarm on startup
        echo "$(date +%s)" > "${LAST_PROGRESS_FILE}"
    fi
fi

send_ntfy "FinQA Monitor" "Started${current_run:+ | ${current_run}}" "low"

# ── Main monitoring loop ──────────────────────────────────────────────────────
tail -n 0 -F "${LOG_FILE}" 2>/dev/null | tr '\r' '\n' | while IFS= read -r line; do

    # ── Detect new run ────────────────────────────────────────────────────────
    if [[ "$line" =~ ^Run:\ thinking=([^,]+),\ model=([^,]+),\ setting=([^,]+),\ split=(.+)$ ]]; then
        short_model="${BASH_REMATCH[2]##*/}"
        current_run="${short_model} ${BASH_REMATCH[3]} think=${BASH_REMATCH[1]}"
        last_notified=0
        # Reset watchdog: model loading can take a few minutes before first tqdm line
        echo "$(date +%s)" > "${LAST_PROGRESS_FILE}"
        log "New run: ${current_run}"
        send_ntfy "FinQA Run Started" "${current_run}" "low"
        continue
    fi

    # ── Track progress (tqdm lines) ───────────────────────────────────────────
    # Format: "Evaluating FinQA:  39%|...| 445/1147 [12:27:54<..."
    if [[ "$line" =~ Evaluating\ FinQA:.*\|\ *([0-9]+)/([0-9]+)\ \[ ]]; then
        current="${BASH_REMATCH[1]}"
        total="${BASH_REMATCH[2]}"

        # Update watchdog timestamp on every tqdm line
        echo "$(date +%s)" > "${LAST_PROGRESS_FILE}"

        if (( current - last_notified >= NOTIFY_EVERY )) || (( current > 0 && current == total )); then
            pct=$(( current * 100 / total ))
            msg="${current}/${total} (${pct}%)"
            [[ -n "$current_run" ]] && msg="${msg}"$'\n'"${current_run}"
            send_ntfy "FinQA Progress" "${msg}"
            log "Sent: ${current}/${total} (${pct}%) | ${current_run}"
            last_notified=$current
        fi
        continue
    fi

    # ── Detect run completion ─────────────────────────────────────────────────
    if [[ "$line" =~ \[saved\]\ (results/[^[:space:]]*/summary\.json) ]]; then
        summary_file="${SCRIPT_DIR}/${BASH_REMATCH[1]}"
        acc_str=""
        if [[ -f "$summary_file" ]]; then
            acc_str=$(python3 - <<EOF 2>/dev/null
import json
with open("${summary_file}") as f:
    data = json.load(f)
runs = data.get("runs", [])
if runs:
    acc = runs[-1].get("accuracy")
    if acc is not None:
        print(f"Acc: {float(acc)*100:.1f}%")
EOF
)
        fi
        msg="Done!${current_run:+ | ${current_run}}"
        [[ -n "$acc_str" ]] && msg="${msg}"$'\n'"${acc_str}"
        send_ntfy "FinQA Run Complete" "${msg}" "high"
        log "Run complete: ${current_run} | ${acc_str}"
        current_run=""
        last_notified=0
        echo "$(date +%s)" > "${LAST_PROGRESS_FILE}"
        continue
    fi

done
