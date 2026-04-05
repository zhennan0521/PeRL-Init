#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PROJECT_DIR}/modules/AReaL"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0


TIMESTAMP=$(date +%Y%m%d%H%M)
LOG_DIR="${PROJECT_DIR}/outputs/logs"
mkdir -p "$LOG_DIR"

EXPERIMENT_NAME=lorafa-1.5b-dapo-async-4node-lr5e-5

python3 "${SCRIPT_DIR}/run_rl.py" \
    --config "${SCRIPT_DIR}/lorafa_async_4node_1.5b_dapo.yaml" \
    experiment_name=${EXPERIMENT_NAME} \
    trial_name=${TIMESTAMP} \
    +scheduler.type=ray \
    cluster.n_nodes=4 \
    cluster.n_gpus_per_node=8 \
    +actor.archon.enable_compile=false \
    "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.log"
