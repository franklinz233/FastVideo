#!/usr/bin/env bash
# Single-node 8-GPU test for validation fix

set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_DIST_INIT_BARRIER_TIMEOUT=3600

export WANDB_API_KEY="${WANDB_API_KEY:-3314e12be1d243355d4ff7b5f7a10995b0c45cbf}"
export WANDB_ENTITY="${WANDB_ENTITY:-1241400738}"
export WANDB_MODE="${WANDB_MODE:-online}"

export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-your_token_here}"
export HF_HOME="/pfs_root/sc/.cache/huggingface"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/pfs_root/sc/outputs/test_8gpu_validation_fix_${TIMESTAMP}"

echo "=== Single-Node 8-GPU Validation Test ==="
echo "Output dir: ${OUTPUT_DIR}"
echo "Testing validation fix at step 600"
echo "=========================================="

torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    --master_addr 127.0.0.1 \
    --master_port 29502 \
    -m fastvideo.train.entrypoint.train \
    --config examples/train/configs/longvid_sft/finetune_wan2.1_spatialvid_1.3B_161f.yaml \
    --training.checkpoint.output_dir "${OUTPUT_DIR}" \
    --training.data.train_batch_size 1 \
    --training.optimizer.learning_rate 5e-7 \
    --callbacks.validation.sampling_steps [50] \
    2>&1 | tee "${OUTPUT_DIR}.log"
