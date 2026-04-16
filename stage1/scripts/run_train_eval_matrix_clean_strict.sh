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

SEED="${SEED:-42}"
EVAL_SETTING="${EVAL_SETTING:-oracle}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:--1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"

MODEL_4B="${MODEL_4B:-Qwen/Qwen3-4B}"
MODEL_8B="${MODEL_8B:-Qwen/Qwen3-8B}"
USE_QLORA_4B="${USE_QLORA_4B:-false}"
USE_QLORA_8B="${USE_QLORA_8B:-true}"

LR_4B="${LR_4B:-5e-5}"
EPOCHS_4B="${EPOCHS_4B:-2}"
LR_8B="${LR_8B:-2e-4}"
EPOCHS_8B="${EPOCHS_8B:-2}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
FULL_MAX_STEPS="${FULL_MAX_STEPS:--1}"

CLEAN_DIR="${CLEAN_DIR:-${STAGE1_ROOT}/data/finqa_clean}"
CLEAN_SUMMARY="${CLEAN_SUMMARY:-${CLEAN_DIR}/clean_summary.json}"
TRAIN_FULL="${TRAIN_FULL:-${CLEAN_DIR}/train_full.jsonl}"
TRAIN_250="${TRAIN_250:-${CLEAN_DIR}/train_250.jsonl}"
TRAIN_1000="${TRAIN_1000:-${CLEAN_DIR}/train_1000.jsonl}"
DEV_FULL="${DEV_FULL:-${CLEAN_DIR}/dev_full.jsonl}"
EXPECTED_TRAIN_KEPT="${EXPECTED_TRAIN_KEPT:-3277}"
EXPECTED_DEV_KEPT="${EXPECTED_DEV_KEPT:-475}"
REBUILD_CLEAN="${REBUILD_CLEAN:-false}"

CONFIG_DIR="${CONFIG_DIR:-${STAGE1_ROOT}/configs/generated_clean_strict}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STAGE1_ROOT}/outputs/sft_clean_strict}"
RESULTS_ROOT="${RESULTS_ROOT:-${BASELINE_ROOT}/results/sft_clean_strict}"
MANIFEST_PATH="${MANIFEST_PATH:-${RESULTS_ROOT}/run_manifest.jsonl}"

mkdir -p "${CONFIG_DIR}" "${OUTPUT_ROOT}" "${RESULTS_ROOT}"
rm -f "${MANIFEST_PATH}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_hf_env() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "HF auth: token present (len=${#HF_TOKEN})"
  else
    log "HF auth: no token found (set HF_TOKEN or HF_TOKEN_FILE)."
  fi
  log "HF cache: HF_HOME=${HF_HOME}"
  log "HF cache: HUB=${HUGGINGFACE_HUB_CACHE}"
  log "HF cache: TRANSFORMERS=${TRANSFORMERS_CACHE}"
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

prefetch_models() {
  log "Prefetching model repos into local cache..."
  local token_arg=()
  if [[ -n "${HF_TOKEN:-}" ]]; then
    token_arg=(--token "${HF_TOKEN}")
  fi
  "${PYTHON_BIN}" "${STAGE1_ROOT}/scripts/prefetch_hf_models.py" \
    --models "${MODEL_4B}" "${MODEL_8B}" \
    --cache_dir "${HF_CACHE_ROOT}" \
    "${token_arg[@]}"
}

resolve_local_model_path() {
  local model="$1"
  "${PYTHON_BIN}" - "${HF_CACHE_ROOT}" "${model}" <<'PY'
import sys
from pathlib import Path

cache_root = Path(sys.argv[1])
model = sys.argv[2]
repo = model.replace("/", "--")
snapshots = cache_root / f"models--{repo}" / "snapshots"
if snapshots.exists():
    cands = [p for p in snapshots.iterdir() if p.is_dir()]
    if cands:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(str(cands[0]))
        raise SystemExit(0)
print(model)
PY
}

sanitize_model() {
  local model="$1"
  echo "${model}" | tr '/:' '__'
}

ensure_clean_data() {
  if [[ "${REBUILD_CLEAN}" == "true" || ! -f "${CLEAN_SUMMARY}" || ! -f "${TRAIN_FULL}" || ! -f "${TRAIN_250}" || ! -f "${TRAIN_1000}" || ! -f "${DEV_FULL}" ]]; then
    log "Rebuilding strict clean splits into ${CLEAN_DIR} ..."
    "${PYTHON_BIN}" "${STAGE1_ROOT}/scripts/build_finqa_clean_splits.py" \
      --output_dir "${CLEAN_DIR}" \
      --seed "${SEED}" \
      --require_scale_relation consistent
  fi

  log "Validating strict clean summary and nested subsets ..."
  "${PYTHON_BIN}" - "${CLEAN_SUMMARY}" "${EXPECTED_TRAIN_KEPT}" "${EXPECTED_DEV_KEPT}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_train = int(sys.argv[2])
expected_dev = int(sys.argv[3])

if not summary_path.exists():
    raise FileNotFoundError(f"missing summary: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
counts = summary.get("counts", {})
train_kept = int(counts.get("train_kept", -1))
dev_kept = int(counts.get("dev_kept", -1))

if train_kept != expected_train:
    raise SystemExit(f"train_kept mismatch: got={train_kept}, expected={expected_train}")
if dev_kept != expected_dev:
    raise SystemExit(f"dev_kept mismatch: got={dev_kept}, expected={expected_dev}")

nested_ok = bool(summary.get("nested_check", {}).get("250_in_1000", False))
if not nested_ok:
    raise SystemExit("nested subset check failed: 250 is not subset of 1000")

print(f"[clean-ok] train_kept={train_kept}, dev_kept={dev_kept}, nested_250_in_1000={nested_ok}")
PY
}

write_config() {
  local run_name="$1"
  local model_name="$2"
  local supervision_style="$3"
  local train_file="$4"
  local use_qlora="$5"
  local learning_rate="$6"
  local num_train_epochs="$7"
  local out_cfg="${CONFIG_DIR}/${run_name}.yaml"

  "${PYTHON_BIN}" - "${STAGE1_ROOT}" "${run_name}" "${model_name}" "${supervision_style}" "${train_file}" "${use_qlora}" "${DEV_FULL}" "${OUTPUT_ROOT}" "${SEED}" "${learning_rate}" "${num_train_epochs}" "${MAX_SEQ_LENGTH}" "${FULL_MAX_STEPS}" "${out_cfg}" <<'PY'
import sys
from pathlib import Path
import yaml

(
    stage1_root,
    run_name,
    model_name,
    supervision_style,
    train_file,
    use_qlora,
    dev_full,
    output_root,
    seed,
    learning_rate,
    num_train_epochs,
    max_seq_length,
    full_max_steps,
    out_cfg,
) = sys.argv[1:]

base_cfg = Path(stage1_root) / "configs" / "full.yaml"
cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
output_dir = Path(output_root) / run_name

cfg["run_name"] = run_name
cfg["seed"] = int(seed)

cfg.setdefault("data", {})
def to_repo_relative(value: str) -> str:
    p = Path(value)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.resolve().relative_to(Path(stage1_root).resolve()))
    except Exception:
        return str(p)

cfg["data"]["train_file"] = to_repo_relative(str(train_file))
cfg["data"]["dev_file"] = to_repo_relative(str(dev_full))
cfg["data"]["max_train_samples"] = None

cfg.setdefault("model", {})
cfg["model"]["model_name_or_path"] = model_name
cfg["model"]["use_qlora"] = str(use_qlora).lower() == "true"

cfg.setdefault("preprocessing", {})
cfg["preprocessing"]["thinking"] = False
cfg["preprocessing"]["supervision_style"] = supervision_style
cfg["preprocessing"]["final_answer_tag"] = "FINAL_ANSWER"

cfg.setdefault("training", {})
cfg["training"]["output_dir"] = to_repo_relative(str(output_dir))
cfg["training"]["learning_rate"] = float(learning_rate)
cfg["training"]["num_train_epochs"] = int(num_train_epochs)
cfg["training"]["max_seq_length"] = int(max_seq_length)
if int(full_max_steps) > 0:
    cfg["training"]["max_steps"] = int(full_max_steps)
else:
    cfg["training"]["max_steps"] = -1

cfg.setdefault("inference", {})
cfg["inference"]["enabled"] = True
cfg["inference"]["checkpoint_dir"] = to_repo_relative(str(output_dir / "checkpoint-last"))
cfg["inference"]["input_file"] = to_repo_relative(str(dev_full))
cfg["inference"]["output_file"] = to_repo_relative(str(output_dir / "infer_predictions.jsonl"))
cfg["inference"]["summary_file"] = to_repo_relative(str(output_dir / "infer_summary.json"))
cfg["inference"]["max_samples"] = 10
cfg["inference"]["mode"] = "dry_run_echo_reference"

cfg.setdefault("logging", {})
cfg["logging"]["log_dir"] = to_repo_relative(str(output_dir / "logs"))
cfg["logging"]["train_log_file"] = to_repo_relative(str(output_dir / "logs" / "train.log"))
cfg["logging"]["infer_log_file"] = to_repo_relative(str(output_dir / "logs" / "infer.log"))

cfg["run_infer_after_train"] = False

Path(out_cfg).parent.mkdir(parents=True, exist_ok=True)
with open(out_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print(out_cfg)
PY
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

  "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${run_name}" "${model_name}" "${model_size}" "${supervision_style}" "${train_size}" "${adapter_path}" "${eval_jsonl}" "${results_dir}" <<'PY'
import json
import sys
from pathlib import Path

(manifest, run_name, model_name, model_size, supervision_style, train_size, adapter_path, eval_jsonl, results_dir) = sys.argv[1:]
payload = {
    "run_name": run_name,
    "phase": "clean_strict",
    "model_name": model_name,
    "model_size": model_size,
    "supervision_style": supervision_style,
    "train_size": train_size,
    "adapter_path": adapter_path,
    "eval_jsonl": eval_jsonl,
    "results_dir": results_dir,
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

  local results_dir="${RESULTS_ROOT}/${run_name}"
  mkdir -p "${results_dir}"

  local model_safe
  model_safe="$(sanitize_model "${model_name}")"
  local eval_jsonl="${results_dir}/finqa_${model_safe}_${EVAL_SETTING}_${EVAL_SPLIT}.jsonl"

  log "Eval run=${run_name} model=${model_name} adapter=${adapter_path} style=${supervision_style}"
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
      --no-enable_thinking
      --max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
      --seed "${SEED}"
      --num_samples "${EVAL_NUM_SAMPLES}"
      --adapter_path "${adapter_path}"
    )
    "${EVAL_PYTHON_BIN}" "${args[@]}"
  )

  append_manifest \
    "${run_name}" "${model_name}" "${model_size}" "${supervision_style}" "${train_size}" \
    "${adapter_path}" "${eval_jsonl}" "${results_dir}"
}

train_and_eval() {
  local run_name="$1"
  local cfg_path="$2"
  local model_name="$3"
  local model_size="$4"
  local supervision_style="$5"
  local train_size="$6"

  log "Train run=${run_name} cfg=${cfg_path}"
  (
    cd "${STAGE1_ROOT}"
    bash scripts/run_train.sh "${cfg_path}"
  )

  local adapter_path="${OUTPUT_ROOT}/${run_name}/checkpoint-last"
  run_eval "${run_name}" "${model_name}" "${model_size}" "${supervision_style}" "${train_size}" "${adapter_path}"
}

run_error_shift_report() {
  log "Running mandatory error-shift analysis ..."
  "${PYTHON_BIN}" "${STAGE1_ROOT}/scripts/analyze_error_shift.py" \
    --manifest_jsonl "${MANIFEST_PATH}" \
    --output_md "${RESULTS_ROOT}/error_shift_report.md" \
    --output_json "${RESULTS_ROOT}/error_shift_report.json"
}

main() {
  log_hf_env
  ensure_eval_dependencies

  log "Stage 0: prefetch local model snapshots"
  prefetch_models
  LOCAL_MODEL_4B="$(resolve_local_model_path "${MODEL_4B}")"
  LOCAL_MODEL_8B="$(resolve_local_model_path "${MODEL_8B}")"
  log "Resolved local cache path 4B: ${LOCAL_MODEL_4B}"
  log "Resolved local cache path 8B: ${LOCAL_MODEL_8B}"
  log "Configs will keep model identifiers: ${MODEL_4B}, ${MODEL_8B}"

  log "Stage 1: strict clean dataset validation"
  ensure_clean_data

  log "Stage 2: launch strict clean 6-run matrix"
  local specs=(
    "4b_clean_full_answer_only|4B|answer_only|full|${TRAIN_FULL}|${LR_4B}|${EPOCHS_4B}"
    "4b_clean_full_formula_rationale|4B|formula_rationale|full|${TRAIN_FULL}|${LR_4B}|${EPOCHS_4B}"
    "8b_clean_full_answer_only|8B|answer_only|full|${TRAIN_FULL}|${LR_8B}|${EPOCHS_8B}"
    "8b_clean_full_formula_rationale|8B|formula_rationale|full|${TRAIN_FULL}|${LR_8B}|${EPOCHS_8B}"
    "8b_clean_250_answer_only|8B|answer_only|250|${TRAIN_250}|${LR_8B}|${EPOCHS_8B}"
    "8b_clean_1000_answer_only|8B|answer_only|1000|${TRAIN_1000}|${LR_8B}|${EPOCHS_8B}"
  )

  local spec run_name model_size style train_size train_file lr epochs
  local model_name model_source use_qlora cfg_path
  for spec in "${specs[@]}"; do
    IFS='|' read -r run_name model_size style train_size train_file lr epochs <<< "${spec}"

    if [[ "${model_size}" == "4B" ]]; then
      model_name="${MODEL_4B}"
      model_source="${MODEL_4B}"
      use_qlora="${USE_QLORA_4B}"
    else
      model_name="${MODEL_8B}"
      model_source="${MODEL_8B}"
      use_qlora="${USE_QLORA_8B}"
    fi

    cfg_path="$(write_config "${run_name}" "${model_source}" "${style}" "${train_file}" "${use_qlora}" "${lr}" "${epochs}")"
    train_and_eval "${run_name}" "${cfg_path}" "${model_name}" "${model_size}" "${style}" "${train_size}"
  done

  log "Stage 3: mandatory error shift"
  run_error_shift_report

  log "All done."
  log "Manifest: ${MANIFEST_PATH}"
  log "Error-shift markdown: ${RESULTS_ROOT}/error_shift_report.md"
}

main "$@"
