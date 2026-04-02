#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/small.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}"
  exit 1
fi

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No Python interpreter found. Install python3 or create .venv first."
  exit 1
fi

# DGX Spark stability defaults (safe to keep on other hosts as well)
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export PYTORCH_NVML_BASED_CUDA_CHECK=1

LOG_FILE=$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = sys.argv[1]

with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

logging_cfg = config.get("logging", {})
training_cfg = config.get("training", {})

log_file = logging_cfg.get("train_log_file")
if not log_file:
    output_dir = training_cfg.get("output_dir", "outputs/default_run")
    log_file = str(Path(output_dir) / "logs" / "train.log")

print(log_file)
PY
)

mkdir -p "$(dirname "${LOG_FILE}")"

echo "========================================"
echo "Running Stage 1 training"
echo "Config file: ${CONFIG_PATH}"
echo "Python bin: ${PYTHON_BIN}"
echo "Train log : ${LOG_FILE}"
echo "========================================"

"${PYTHON_BIN}" -u train_sft.py --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"
