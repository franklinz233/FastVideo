#!/bin/bash
# Launch Wan 2.1 1.3B training on Sekai (321-frame, ~20s) across multiple nodes
# Stage 3 of curriculum learning: resume from 161f checkpoint
# Usage: RANK=<node_rank> WORLD_SIZE=<num_nodes> [RESUME_FROM=<ckpt_dir>] bash scripts/longvid_sft/longvid_sft_train_321f_multinode.sh
# Example (8 nodes, 64 GPUs total):
#   Node 0: RANK=0 WORLD_SIZE=8 RESUME_FROM=/pfs_root/sc/outputs/longvid_sft_wan2.1_1.3B_161f_8x8_20250101_130000 bash scripts/longvid_sft/longvid_sft_train_321f_multinode.sh
#   Node 1: RANK=1 WORLD_SIZE=8 RESUME_FROM=/pfs_root/sc/outputs/longvid_sft_wan2.1_1.3B_161f_8x8_20250101_130000 bash scripts/longvid_sft/longvid_sft_train_321f_multinode.sh

source "$(dirname "$0")/env.sh"

NODE_RANK=${RANK:-0}
NNODES=${WORLD_SIZE:-8}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG="examples/train/configs/longvid_sft/finetune_wan2.1_longvid_sft_1.3B_321f_from161fshift5.yaml"
COORD_DIR="/pfs_root/sc/torch_coord"
CONFIG_NAME="longvid_sft_1.3B_321f_from161fshift5"

mkdir -p "$COORD_DIR"

echo "=========================================="
echo "Config: $CONFIG"
echo "Node Rank: $NODE_RANK / $NNODES nodes"
echo "=========================================="

# ===== Master 地址协商 =====
if [ "$NODE_RANK" -eq 0 ]; then
    JOB_ID=$(openssl rand -hex 3 2>/dev/null || echo $RANDOM)
    MASTER_FILE="$COORD_DIR/${CONFIG_NAME}.${JOB_ID}.master"
    MY_IP=$(hostname -I | awk '{print $1}')
    RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "${MY_IP} ${RUN_TIMESTAMP}" > "$MASTER_FILE"
    echo "[INFO] Master IP: $MY_IP, beacon: $MASTER_FILE"
    echo "[INFO] Run timestamp: $RUN_TIMESTAMP"
    MASTER_ADDR="$MY_IP"
else
    echo "[INFO] Rank $NODE_RANK: waiting for master file..."
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
                echo "[SUCCESS] Master IP: $MASTER_ADDR"
                echo "[INFO] Run timestamp: $RUN_TIMESTAMP"
                break
            fi
        fi
        [ $((i % 5)) -eq 0 ] && echo "[WAIT] scanning... ($i/600s)"
        sleep 1
    done
    if [ -z "$MASTER_ADDR" ]; then
        echo "[ERROR] Timeout: master file not found"
        exit 1
    fi
fi

OUTPUT_DIR="/pfs_root/sc/outputs/${CONFIG_NAME}_${NNODES}x8_${RUN_TIMESTAMP}"
echo "[INFO] Output dir: $OUTPUT_DIR"

# ===== Resume checkpoint (optional) =====
RESUME_ARGS=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS="--training.checkpoint.resume_from_checkpoint $RESUME_FROM"
    echo "[INFO] Resuming from: $RESUME_FROM"
fi

# ===== 启动训练 =====
NUM_GPUS=$((NNODES * 8))
echo "[INFO] Total GPUs: $NUM_GPUS (${NNODES} nodes x 8)"

torchrun \
    --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m fastvideo.train.entrypoint.train \
    --config $CONFIG \
    --training.distributed.num_gpus $NUM_GPUS \
    --training.distributed.hsdp_replicate_dim $((NUM_GPUS / 4)) \
    --training.checkpoint.output_dir $OUTPUT_DIR \
    $RESUME_ARGS

# ===== 清理 =====
if [ "$NODE_RANK" -eq 0 ]; then
    rm -f "$MASTER_FILE"
    echo "[INFO] Master file cleaned up."
fi
