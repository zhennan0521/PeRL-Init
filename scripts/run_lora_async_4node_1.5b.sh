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

# ============================================================
# 4-Node Ray Launch for LoRA Async Training
# ============================================================
# Prerequisites:
#   1. Start Ray head on node 0:
#      ray start --head --port=6379
#
#   2. Join workers on node 1/2/3:
#      ray start --address=<head_ip>:6379
#
#   3. Verify cluster:
#      ray status
#
#   4. Run this script on the head node.
# ============================================================

python3 -m areal.infra.launcher.ray \
    --config "${SCRIPT_DIR}/lora_async_4node_1.5b_dapo.yaml" \
    experiment_name=lora-1.5b-dapo-async-4node-lr5e-5 \
    trial_name=${TIMESTAMP} \
    +allocation_mode=sglang:d24p1t1+archon:d8p1t1 \
    cluster.n_nodes=4 \
    cluster.n_gpus_per_node=8 \
    "$@" 2>&1 | tee "${LOG_DIR}/lora_async_4node_1.5b_${TIMESTAMP}.log"
