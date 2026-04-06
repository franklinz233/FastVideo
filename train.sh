#!/bin/bash

LOG_DIR="/root/data_root/outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_rank${RANK:-0}.log"

echo "[INFO] Logging to: $LOG_FILE"

bash examples/train/run_multinode_simple.sh \
examples/train/configs/longvid_sft/finetune_wan2.1_spatialvid_1.3B_161f.yaml \
--training.distributed.hsdp_replicate_dim 8 \
--training.distributed.hsdp_shard_dim 4 \
--training.data.train_batch_size 4 \
--training.optimizer.learning_rate 1.6e-6 \
--training.data.dataloader_num_workers 8 \
2>&1 | tee "$LOG_FILE"