#!/usr/bin/env bash
# General-purpose multi-node training launcher for clusters without Slurm.
#
# Master address is negotiated via a shared filesystem (no direct IP discovery
# needed). The cluster scheduler runs this script on every node and injects
# RANK (node index 0,1,2…) and WORLD_SIZE (total nodes).
#
# Usage:
#   bash examples/train/run_multinode.sh <config.yaml> [--dotted.key value ...]
#
# Examples:
#   bash examples/train/run_multinode.sh \
#       examples/train/configs/sekai/finetune_wan2.1_spatialvid_1.3B_161f.yaml
#
#   bash examples/train/run_multinode.sh \
#       examples/train/configs/sekai/finetune_wan2.1_spatialvid_1.3B_161f.yaml \
#       --training.checkpoint.resume_from_checkpoint /path/to/ckpt
#
# Environment variables (injected by scheduler or overridable):
#   RANK          Node rank (0-based)                     [REQUIRED]
#   WORLD_SIZE    Total number of nodes                   [REQUIRED]
#   NUM_GPUS      GPUs per node                           (default: 8)
#   MASTER_PORT   Rendezvous port                         (default: 29500)
#   COORD_DIR     Shared dir for master addr negotiation  (default: /pfs_root/sc/torch_coord)

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
COORD_DIR="${COORD_DIR:-/pfs_root/sc/torch_coord}"

TOTAL_GPUS=$((NNODES * NUM_GPUS))
CONFIG_NAME="$(basename "${CONFIG}" .yaml)"

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
echo "Extra args: ${EXTRA_ARGS[*]:-}"
echo "=========================================="

# ── Master address negotiation via shared filesystem ─────────────────
mkdir -p "$COORD_DIR"

if [ "$NODE_RANK" -eq 0 ]; then
    # Rank 0: write IP + timestamp to beacon file
    JOB_ID=$(openssl rand -hex 3 2>/dev/null || echo $RANDOM)
    MASTER_FILE="$COORD_DIR/${CONFIG_NAME}.${JOB_ID}.master"
    MY_IP=$(hostname -I | awk '{print $1}')
    RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "${MY_IP} ${RUN_TIMESTAMP}" > "$MASTER_FILE"
    MASTER_ADDR="$MY_IP"
    echo "[INFO] Master IP: $MASTER_ADDR, beacon: $MASTER_FILE"
    echo "[INFO] Run timestamp: $RUN_TIMESTAMP"

    # Cleanup beacon file on exit (success, error, or signal)
    cleanup() { rm -f "$MASTER_FILE"; echo "[INFO] Master beacon cleaned up."; }
    trap cleanup EXIT
else
    # Other ranks: poll for master beacon file
    echo "[INFO] Rank $NODE_RANK: waiting for master beacon..."
    FILE_PATTERN="${CONFIG_NAME}.*.master"
    MASTER_ADDR=""
    for i in $(seq 1 600); do
        FOUND_FILE=$(ls $COORD_DIR/$FILE_PATTERN 2>/dev/null | head -1)
        if [ -n "$FOUND_FILE" ]; then
            CONTENT=$(cat "$FOUND_FILE")
            CANDIDATE_IP=$(echo "$CONTENT" | awk '{print $1}')
            RUN_TIMESTAMP=$(echo "$CONTENT" | awk '{print $2}')
            if [[ "$CANDIDATE_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                MASTER_ADDR="$CANDIDATE_IP"
                echo "[SUCCESS] Master IP: $MASTER_ADDR (from $FOUND_FILE)"
                echo "[INFO] Run timestamp: $RUN_TIMESTAMP"
                break
            fi
        fi
        [ $((i % 10)) -eq 0 ] && echo "[WAIT] scanning for master beacon... ($i/600s)"
        sleep 1
    done
    if [ -z "$MASTER_ADDR" ]; then
        echo "[ERROR] Timeout: master beacon not found after 600s"
        exit 1
    fi
fi

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
