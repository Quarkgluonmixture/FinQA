#!/usr/bin/env bash
set -euo pipefail

# Apply machine-specific runtime environment required by:
# /home/jiaming/workspace/FinQA/DGX_SPARK_MACHINE_QUIRKS.md
#
# Usage:
#   source scripts/apply_dgx_spark_quirks.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QUIRKS_DOC="${ROOT_DIR}/DGX_SPARK_MACHINE_QUIRKS.md"

if [[ ! -f "${QUIRKS_DOC}" ]]; then
  echo "[warn] DGX quirks doc not found: ${QUIRKS_DOC}" >&2
fi

# Required on spark-9ea3 to avoid CUDA init stalls/hangs.
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export PYTORCH_NVML_BASED_CUDA_CHECK=1

# Required on this host to avoid XET 416 shard download failures.
export HF_HUB_DISABLE_XET=1

