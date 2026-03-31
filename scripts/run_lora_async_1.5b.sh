#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PROJECT_DIR}/modules/AReaL"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=WARN

TIMESTAMP=$(date +%Y%m%d%H%M)
LOG_DIR="${PROJECT_DIR}/outputs/logs"
mkdir -p "$LOG_DIR"

python3 "${SCRIPT_DIR}/run_rl.py" \
    --config "${SCRIPT_DIR}/lora_async_1.5b_dapo.yaml" \
    experiment_name=lora-1.5b-dapo-async \
    trial_name=${TIMESTAMP} \
    +actor.archon.enable_compile=false \
    "$@" 2>&1 | tee "${LOG_DIR}/lora_1.5b_dapo_${TIMESTAMP}.log"
