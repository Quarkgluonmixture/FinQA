#!/usr/bin/env bash
set -euo pipefail

# Phase 6: Re-evaluate 8B SFT checkpoints with chat_default prompt protocol
# + rebuild error_shift report for Phase 5 prompt-aligned results
#
# This script:
# 1. Evaluates 4 x 8B SFT checkpoints using chat_default (original chat format)
# 2. Rebuilds manifest and runs error_shift for prompt_aligned results (N2)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_ROOT="${STAGE1_ROOT}/../finqa_baseline"

# ---------- Python ----------
if [[ -x "${BASELINE_ROOT}/.venv/bin/python" ]]; then
  EVAL_PYTHON_BIN="${BASELINE_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  EVAL_PYTHON_BIN="$(command -v python3)"
else
  echo "[error] no usable python found." >&2; exit 127
fi

if [[ -x "${STAGE1_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${STAGE1_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="${EVAL_PYTHON_BIN}"
fi

# ---------- DGX quirks ----------
if [[ -f "${BASELINE_ROOT}/scripts/apply_dgx_spark_quirks.sh" ]]; then
  source "${BASELINE_ROOT}/scripts/apply_dgx_spark_quirks.sh"
fi

# ---------- HF cache ----------
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

# ---------- Eval params ----------
SEED="${SEED:-42}"
EVAL_SETTING="${EVAL_SETTING:-oracle}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:--1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
MODEL_8B="${MODEL_8B:-Qwen/Qwen3-8B}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${STAGE1_ROOT}/outputs/sft_clean_strict}"

# Task A results
CHAT_RESULTS_ROOT="${CHAT_RESULTS_ROOT:-${BASELINE_ROOT}/results/sft_clean_strict_chat_eval}"
CHAT_MANIFEST="${CHAT_RESULTS_ROOT}/run_manifest.jsonl"

# Task B (error shift for prompt-aligned)
PA_RESULTS_ROOT="${PA_RESULTS_ROOT:-${BASELINE_ROOT}/results/sft_clean_strict_prompt_aligned_eval}"
PA_MANIFEST="${PA_RESULTS_ROOT}/run_manifest.jsonl"
PA_SFT_MANIFEST="${PA_RESULTS_ROOT}/run_manifest_sft_only.jsonl"

EVAL_RETRY_MAX="${EVAL_RETRY_MAX:-5}"
EVAL_RETRY_SLEEP="${EVAL_RETRY_SLEEP:-30}"

mkdir -p "${CHAT_RESULTS_ROOT}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

sanitize_model() { echo "$1" | tr '/:' '__'; }

# ---------- run_eval ----------
run_eval() {
  local run_name="$1" model_name="$2" adapter_path="$3" results_dir="$4" prompt_protocol="$5" manifest="$6"

  mkdir -p "${results_dir}"
  local model_safe
  model_safe="$(sanitize_model "${model_name}")"
  local eval_jsonl="${results_dir}/finqa_${model_safe}_${EVAL_SETTING}_${EVAL_SPLIT}.jsonl"

  # Skip if summary.json already exists (resume-safe)
  if [[ -f "${results_dir}/summary.json" ]]; then
    log "[skip] ${run_name} already has summary.json, skipping."
    # Still append to manifest
    _append_manifest "${manifest}" "${run_name}" "${model_name}" "${adapter_path}" "${eval_jsonl}" "${results_dir}" "${prompt_protocol}"
    return 0
  fi

  log "Eval run=${run_name} model=${model_name} adapter=${adapter_path:-<none>} protocol=${prompt_protocol}"

  local attempt=1 rc=0
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
        --prompt_protocol "${prompt_protocol}"
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

    if [[ ${rc} -eq 0 ]]; then break; fi
    if [[ ${attempt} -ge ${EVAL_RETRY_MAX} ]]; then
      log "[error] Eval run=${run_name} failed after ${attempt} attempts (exit=${rc})."
      return "${rc}"
    fi
    log "[warn] Eval run=${run_name} failed (exit=${rc}); retry after ${EVAL_RETRY_SLEEP}s."
    sleep "${EVAL_RETRY_SLEEP}"
    attempt=$((attempt + 1))
  done

  _append_manifest "${manifest}" "${run_name}" "${model_name}" "${adapter_path}" "${eval_jsonl}" "${results_dir}" "${prompt_protocol}"
}

_append_manifest() {
  local manifest="$1" run_name="$2" model_name="$3" adapter_path="$4" eval_jsonl="$5" results_dir="$6" prompt_protocol="$7"
  "${PYTHON_BIN}" - "${manifest}" "${run_name}" "${model_name}" "${adapter_path}" "${eval_jsonl}" "${results_dir}" "${prompt_protocol}" <<'PY'
import json, sys
from pathlib import Path
manifest, run_name, model_name, adapter_path, eval_jsonl, results_dir, prompt_protocol = sys.argv[1:]
payload = {
    "run_name": run_name,
    "model_name": model_name,
    "adapter_path": adapter_path if adapter_path else "",
    "eval_jsonl": eval_jsonl,
    "results_dir": results_dir,
    "prompt_protocol": prompt_protocol,
    "supervision_style": "answer_only" if "answer_only" in run_name else ("formula_rationale" if "formula_rationale" in run_name else "zero_shot"),
    "model_size": "4B" if "4b" in run_name.lower() or "4B" in model_name else "8B",
    "train_size": "250" if "250" in run_name else ("1000" if "1000" in run_name else ("0" if "zeroshot" in run_name else "full")),
    "is_sft": "zeroshot" not in run_name,
}
p = Path(manifest)
p.parent.mkdir(parents=True, exist_ok=True)
with p.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
}

# ============================================================
# TASK B: Rebuild error shift for prompt-aligned Phase 5
# ============================================================
run_task_b() {
  log "=== Task B: Rebuild error shift for prompt-aligned eval ==="

  # Rebuild manifest from existing summary.json files
  rm -f "${PA_MANIFEST}" "${PA_SFT_MANIFEST}"

  local pa_runs=(
    "4b_clean_full_answer_only:Qwen/Qwen3-4B:${CHECKPOINT_ROOT}/4b_clean_full_answer_only/checkpoint-last"
    "4b_clean_full_formula_rationale:Qwen/Qwen3-4B:${CHECKPOINT_ROOT}/4b_clean_full_formula_rationale/checkpoint-last"
    "8b_clean_full_answer_only:Qwen/Qwen3-8B:${CHECKPOINT_ROOT}/8b_clean_full_answer_only/checkpoint-last"
    "8b_clean_full_formula_rationale:Qwen/Qwen3-8B:${CHECKPOINT_ROOT}/8b_clean_full_formula_rationale/checkpoint-last"
    "8b_clean_250_answer_only:Qwen/Qwen3-8B:${CHECKPOINT_ROOT}/8b_clean_250_answer_only/checkpoint-last"
    "8b_clean_1000_answer_only:Qwen/Qwen3-8B:${CHECKPOINT_ROOT}/8b_clean_1000_answer_only/checkpoint-last"
    "4b_zeroshot_prompt_aligned:Qwen/Qwen3-4B:"
    "8b_zeroshot_prompt_aligned:Qwen/Qwen3-8B:"
  )

  for entry in "${pa_runs[@]}"; do
    IFS=: read -r rname model adapter <<< "${entry}"
    local rdir="${PA_RESULTS_ROOT}/${rname}"
    local model_safe
    model_safe="$(sanitize_model "${model}")"
    local jsonl="${rdir}/finqa_${model_safe}_${EVAL_SETTING}_${EVAL_SPLIT}.jsonl"
    if [[ -f "${rdir}/summary.json" ]]; then
      _append_manifest "${PA_MANIFEST}" "${rname}" "${model}" "${adapter}" "${jsonl}" "${rdir}" "stage1_train_text"
    else
      log "[warn] Missing summary.json for ${rname}, skipping manifest entry."
    fi
  done

  # Build SFT-only manifest
  "${PYTHON_BIN}" - "${PA_MANIFEST}" "${PA_SFT_MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
manifest_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
with manifest_path.open("r") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        row = json.loads(line)
        if row.get("supervision_style") in {"answer_only", "formula_rationale"}:
            rows.append(row)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[saved] {out_path} ({len(rows)} entries)")
PY

  # Run error shift analysis
  "${PYTHON_BIN}" "${STAGE1_ROOT}/scripts/analyze_error_shift.py" \
    --manifest_jsonl "${PA_SFT_MANIFEST}" \
    --output_md "${PA_RESULTS_ROOT}/error_shift_report.md" \
    --output_json "${PA_RESULTS_ROOT}/error_shift_report.json"

  log "Task B complete. Error shift report: ${PA_RESULTS_ROOT}/error_shift_report.md"
}

# ============================================================
# TASK A: 8B SFT chat-format re-evaluation
# ============================================================
run_task_a() {
  log "=== Task A: 8B SFT chat-format re-evaluation (4 runs) ==="

  rm -f "${CHAT_MANIFEST}"

  local ckpt_8b_answer="${CHECKPOINT_ROOT}/8b_clean_full_answer_only/checkpoint-last"
  local ckpt_8b_formula="${CHECKPOINT_ROOT}/8b_clean_full_formula_rationale/checkpoint-last"
  local ckpt_8b_250="${CHECKPOINT_ROOT}/8b_clean_250_answer_only/checkpoint-last"
  local ckpt_8b_1000="${CHECKPOINT_ROOT}/8b_clean_1000_answer_only/checkpoint-last"

  for ckpt_dir in "${ckpt_8b_answer}" "${ckpt_8b_formula}" "${ckpt_8b_250}" "${ckpt_8b_1000}"; do
    if [[ ! -d "${ckpt_dir}" ]]; then
      log "[error] Missing checkpoint: ${ckpt_dir}"
      exit 1
    fi
  done

  run_eval "8b_clean_full_answer_only" "${MODEL_8B}" "${ckpt_8b_answer}" \
    "${CHAT_RESULTS_ROOT}/8b_clean_full_answer_only" "chat_default" "${CHAT_MANIFEST}"

  run_eval "8b_clean_full_formula_rationale" "${MODEL_8B}" "${ckpt_8b_formula}" \
    "${CHAT_RESULTS_ROOT}/8b_clean_full_formula_rationale" "chat_default" "${CHAT_MANIFEST}"

  run_eval "8b_clean_250_answer_only" "${MODEL_8B}" "${ckpt_8b_250}" \
    "${CHAT_RESULTS_ROOT}/8b_clean_250_answer_only" "chat_default" "${CHAT_MANIFEST}"

  run_eval "8b_clean_1000_answer_only" "${MODEL_8B}" "${ckpt_8b_1000}" \
    "${CHAT_RESULTS_ROOT}/8b_clean_1000_answer_only" "chat_default" "${CHAT_MANIFEST}"

  log "Task A complete. Results: ${CHAT_RESULTS_ROOT}/"

  # Also run error shift on chat eval results
  local chat_sft_manifest="${CHAT_RESULTS_ROOT}/run_manifest_sft_only.jsonl"
  cp "${CHAT_MANIFEST}" "${chat_sft_manifest}"
  "${PYTHON_BIN}" "${STAGE1_ROOT}/scripts/analyze_error_shift.py" \
    --manifest_jsonl "${chat_sft_manifest}" \
    --output_md "${CHAT_RESULTS_ROOT}/error_shift_report.md" \
    --output_json "${CHAT_RESULTS_ROOT}/error_shift_report.json"

  log "Chat eval error shift report: ${CHAT_RESULTS_ROOT}/error_shift_report.md"
}

# ============================================================
# MAIN
# ============================================================
main() {
  log "=========================================="
  log "Phase 6: 8B chat eval + error shift rebuild"
  log "=========================================="

  # Task B first (fast, ~10 min)
  run_task_b

  # Task A (slow, ~1 day)
  run_task_a

  log "All done."
}

main "$@"
