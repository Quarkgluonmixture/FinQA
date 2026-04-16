#!/usr/bin/env bash
set -euo pipefail

echo "[deprecated] run_forever_no_guard.sh has been renamed to run_verification_matrix_loop.sh" >&2
echo "[deprecated] This wrapper will be removed in a future cleanup cycle." >&2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_verification_matrix_loop.sh" "$@"
