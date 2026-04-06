#!/bin/bash
# Common environment for all longvid_sft/spatialvid scripts.
# Usage: source "$(dirname "$0")/env.sh"   (from same-dir scripts)
#        source "$(cd "$(dirname "$0")/.." && pwd)/longvid_sft/env.sh"  (if needed)

# ===== Paths =====
export DATA_ROOT="/root/data_root"
export PFS_ROOT="/pfs_root/sc"
export CODE_DIR="$PFS_ROOT/code/FastVideo"
export HF_HOME="$PFS_ROOT/.cache/huggingface"
export COORD_DIR="$PFS_ROOT/torch_coord"

# ===== HuggingFace =====
export HF_ENDPOINT="https://hf-mirror.com"
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-your_token_here}"

# ===== WandB =====
export WANDB_API_KEY="3314e12be1d243355d4ff7b5f7a10995b0c45cbf"
export WANDB_ENTITY="1241400738"
export WANDB_MODE="${WANDB_MODE:-online}"

# ===== NCCL =====
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# ===== Misc =====
export TOKENIZERS_PARALLELISM=false

# ===== Conda + working directory =====
source /pfs_root/sc/miniconda3/bin/activate fastvideo
cd "$CODE_DIR"
