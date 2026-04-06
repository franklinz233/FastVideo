#!/usr/bin/env bash
# Launch training from a YAML config.
#
# Usage:
#   bash examples/train/run.sh <config.yaml> [--dotted.key value ...]
#
# Examples:
#   bash examples/train/run.sh examples/train/configs/fine_tuning/wan/t2v.yaml
#   bash examples/train/run.sh examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml --dry-run
#   bash examples/train/run.sh examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml \
#       --training.distributed.num_gpus 4 \
#       --training.optimizer.learning_rate 1e-5
#   bash examples/train/run.sh examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml \
#       --training.checkpoint.resume_from_checkpoint outputs/my_run/checkpoint-1000
#
# Logs are written to logs/<config_name>_<timestamp>.log (and also printed to stdout).

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [extra flags...]}"
shift

# ── GPU / node settings ──────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NUM_GPUS="${NUM_GPUS:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29502}"
export TOKENIZERS_PARALLELISM=false
# ── W&B ──────────────────────────────────────────────────────────
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_DIST_INIT_BARRIER_TIMEOUT=3600
export TOKENIZERS_PARALLELISM=false

export WANDB_API_KEY="${WANDB_API_KEY:-3314e12be1d243355d4ff7b5f7a10995b0c45cbf}"
export WANDB_ENTITY="${WANDB_ENTITY:-1241400738}"
export WANDB_MODE="${WANDB_MODE:-online}"

# export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-your_token_here}"
export HF_HOME="/pfs_root/sc/.cache/huggingface"


# ── Log file ─────────────────────────────────────────────────────
CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-examples/train}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${TIMESTAMP}.log"

echo "=== Train Training ==="
echo "Config:      ${CONFIG}"
echo "Num GPUs:    ${NUM_GPUS}"
echo "Num Nodes:   ${NNODES}"
echo "Node Rank:   ${NODE_RANK}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "Extra args:  $*"
echo "Log file:    ${LOG_FILE}"
echo "=============================="


torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${NUM_GPUS}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    -m fastvideo.train.entrypoint.train \
    --config "${CONFIG}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
