#!/usr/bin/env bash
# Simplified multi-node training launcher for clusters with pre-configured MASTER_ADDR.
#
# Usage:
#   bash examples/train/run_multinode_simple.sh <config.yaml> [--dotted.key value ...]
#
# Environment variables (set by cluster scheduler):
#   RANK          Node rank (0-based)                     [REQUIRED]
#   WORLD_SIZE    Total number of nodes                   [REQUIRED]
#   MASTER_ADDR   Master node address                     [auto-detected if not set]
#   NUM_GPUS      GPUs per node                           (default: 8)
#   MASTER_PORT   Rendezvous port                         (default: 29500)

set -euo pipefail

# ── Arguments ───────────────────────────────────────────────────────
CONFIG="${1:?Usage: $0 <config.yaml> [extra flags...]}"
shift
EXTRA_ARGS=("$@")

# ── Environment defaults ────────────────────────────────────────────
NODE_RANK="${RANK:?RANK env var is required (set by scheduler)}"
NNODES="${WORLD_SIZE:?WORLD_SIZE env var is required (set by scheduler)}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ── Master address detection ────────────────────────────────────────
if [ -z "${MASTER_ADDR:-}" ]; then
    echo "[INFO] MASTER_ADDR not set, using hostname resolution..."
    if [ "$NODE_RANK" -eq 0 ]; then
        # Rank 0: use own hostname
        MASTER_ADDR=$(hostname)
        echo "[INFO] Master node, using hostname: $MASTER_ADDR"
    else
        echo "[ERROR] MASTER_ADDR must be set by scheduler for worker nodes"
        exit 1
    fi
else
    echo "[INFO] Using MASTER_ADDR from environment: $MASTER_ADDR"
fi

TOTAL_GPUS=$((NNODES * NUM_GPUS))
CONFIG_NAME="$(basename "${CONFIG}" .yaml)"

# ── Synchronize run timestamp across nodes ──────────────────────────
# Each node calling `date` independently produces slightly different
# timestamps, causing DCP shards to land in different output dirs.
# Solution: Node 0 writes a timestamp beacon to shared storage;
# worker nodes wait to read it.
COORD_DIR="${COORD_DIR:-/pfs_root/sc/torch_coord}"
mkdir -p "$COORD_DIR"
_TS_BEACON="$COORD_DIR/${CONFIG_NAME}.run_ts"

if [ "$NODE_RANK" -eq 0 ]; then
    RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "$RUN_TIMESTAMP" > "$_TS_BEACON"
    echo "[INFO] Wrote run timestamp beacon: $RUN_TIMESTAMP"
else
    echo "[INFO] Rank $NODE_RANK: waiting for timestamp beacon..."
    for _i in $(seq 1 120); do
        if [ -f "$_TS_BEACON" ]; then
            RUN_TIMESTAMP=$(cat "$_TS_BEACON" | tr -d '[:space:]')
            if [ -n "$RUN_TIMESTAMP" ]; then
                echo "[INFO] Got run timestamp from beacon: $RUN_TIMESTAMP"
                break
            fi
        fi
        sleep 1
    done
    if [ -z "${RUN_TIMESTAMP:-}" ]; then
        echo "[ERROR] Timeout waiting for timestamp beacon"
        exit 1
    fi
fi

# ── Cluster environment ─────────────────────────────────────────────
source /pfs_root/sc/miniconda3/bin/activate fastvideo

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_DIST_INIT_BARRIER_TIMEOUT=3600
export TOKENIZERS_PARALLELISM=false

export WANDB_API_KEY="${WANDB_API_KEY:-3314e12be1d243355d4ff7b5f7a10995b0c45cbf}"
export WANDB_ENTITY="${WANDB_ENTITY:-1241400738}"
export WANDB_MODE="${WANDB_MODE:-online}"

export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-your_token_here}"
export HF_HOME="/pfs_root/sc/.cache/huggingface"

echo "=========================================="
echo "Config:     $CONFIG"
echo "Node Rank:  $NODE_RANK / $NNODES nodes"
echo "GPUs/node:  $NUM_GPUS"
echo "Total GPUs: $TOTAL_GPUS"
echo "Master:     $MASTER_ADDR:$MASTER_PORT"
echo "Extra args: ${EXTRA_ARGS[*]:-}"
echo "=========================================="

# ── Output directory ────────────────────────────────────────────────
OUTPUT_DIR="/pfs_root/sc/outputs/${CONFIG_NAME}_${NNODES}x${NUM_GPUS}_${RUN_TIMESTAMP}"
echo "[INFO] Output dir: $OUTPUT_DIR"

# ── Launch training ─────────────────────────────────────────────────
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m fastvideo.train.entrypoint.train \
    --config $CONFIG \
    --training.distributed.num_gpus $TOTAL_GPUS \
    --training.checkpoint.output_dir $OUTPUT_DIR \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# ── Cleanup timestamp beacon ────────────────────────────────────────
if [ "$NODE_RANK" -eq 0 ] && [ -f "$_TS_BEACON" ]; then
    rm -f "$_TS_BEACON"
    echo "[INFO] Timestamp beacon cleaned up."
fi
