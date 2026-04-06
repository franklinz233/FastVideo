#!/bin/bash
# Multi-node preprocessing for SpatialVID-HQ 161f dataset.
# Runs on all nodes via cluster scheduler; RANK and WORLD_SIZE are injected.
#
# Usage (run on each node):
#   RANK=0 WORLD_SIZE=4 bash spatialvid_preprocess_161f_multinode.sh
#   RANK=1 WORLD_SIZE=4 bash spatialvid_preprocess_161f_multinode.sh
#   ...
#
# Environment variables:
#   RANK          Node rank (0-based)                     [REQUIRED]
#   WORLD_SIZE    Total number of nodes                   [REQUIRED]
#   NUM_GPUS      GPUs per node                           (default: 8)
#   MASTER_PORT   Rendezvous port                         (default: 29510)

set -euo pipefail

source "$(dirname "$0")/env.sh"

# ── Config ───────────────────────────────────────────────────────────
NODE_RANK="${RANK:?RANK env var is required}"
NNODES="${WORLD_SIZE:?WORLD_SIZE env var is required}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29510}"
TOTAL_GPUS=$((NNODES * NUM_GPUS))

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATASET_PATH="/pfs_root/sc/dataset/SpatialVID-HQ/formatted-161f/"
OUTPUT_DIR="/pfs_root/sc/dataset/SpatialVID-HQ/processed-161f-32gpu/"

echo "=========================================="
echo "Node Rank:  $NODE_RANK / $NNODES nodes"
echo "GPUs/node:  $NUM_GPUS"
echo "Total GPUs: $TOTAL_GPUS"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

# ── Master address negotiation via shared filesystem ─────────────────
JOB_NAME="spatialvid_preprocess_161f"
mkdir -p "$COORD_DIR"

if [ "$NODE_RANK" -eq 0 ]; then
    JOB_ID=$(openssl rand -hex 3 2>/dev/null || echo $RANDOM)
    MASTER_FILE="$COORD_DIR/${JOB_NAME}.${JOB_ID}.master"
    MY_IP=$(hostname -I | awk '{print $1}')
    echo "${MY_IP}" > "$MASTER_FILE"
    MASTER_ADDR="$MY_IP"
    echo "[INFO] Master IP: $MASTER_ADDR, beacon: $MASTER_FILE"
    cleanup() { rm -f "$MASTER_FILE"; }
    trap cleanup EXIT
else
    echo "[INFO] Rank $NODE_RANK: waiting for master beacon..."
    FILE_PATTERN="${JOB_NAME}.*.master"
    MASTER_ADDR=""
    for i in $(seq 1 300); do
        FOUND_FILE=$(ls $COORD_DIR/$FILE_PATTERN 2>/dev/null | head -1)
        if [ -n "$FOUND_FILE" ]; then
            CANDIDATE_IP=$(cat "$FOUND_FILE" | awk '{print $1}')
            if [[ "$CANDIDATE_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                MASTER_ADDR="$CANDIDATE_IP"
                echo "[SUCCESS] Master IP: $MASTER_ADDR"
                break
            fi
        fi
        [ $((i % 10)) -eq 0 ] && echo "[WAIT] scanning... ($i/300s)"
        sleep 1
    done
    if [ -z "$MASTER_ADDR" ]; then
        echo "[ERROR] Timeout: master beacon not found after 300s"
        exit 1
    fi
fi

# ── Launch ───────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path $MODEL_PATH \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.video_loader_type torchvision \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR \
    --preprocess.preprocess_video_batch_size 2 \
    --preprocess.dataloader_num_workers 4 \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 161 \
    --preprocess.train_fps 16 \
    --preprocess.samples_per_file 8 \
    --preprocess.flush_frequency 8 \
    --preprocess.video_length_tolerance_range 15
