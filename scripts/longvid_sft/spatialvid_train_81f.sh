#!/bin/bash
# Launch Wan 2.1 1.3B training on SpatialVID-HQ dataset (81-frame clips)
# num_latent_t=21, 480x832, sp_size=1 (short sequence, no sequence parallelism needed)
# Usage: [RESUME_FROM=<ckpt_dir>] bash scripts/longvid_sft/spatialvid_train_81f.sh

source "$(dirname "$0")/env.sh"

MASTER_PORT=${MASTER_PORT:-29502}
CONFIG="examples/train/configs/longvid_sft/finetune_wan2.1_spatialvid_1.3B_81f.yaml"
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/pfs_root/sc/outputs/spatialvid_wan2.1_1.3B_81f_1x8_${RUN_TIMESTAMP}"

RESUME_ARGS=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS="--training.checkpoint.resume_from_checkpoint $RESUME_FROM"
    echo "[INFO] Resuming from: $RESUME_FROM"
fi

echo "=========================================="
echo "Config : $CONFIG"
echo "Output : $OUTPUT_DIR"
echo "=========================================="

torchrun \
    --nproc_per_node=8 \
    --master_port=$MASTER_PORT \
    -m fastvideo.train.entrypoint.train \
    --config $CONFIG \
    --training.checkpoint.output_dir $OUTPUT_DIR \
    $RESUME_ARGS
