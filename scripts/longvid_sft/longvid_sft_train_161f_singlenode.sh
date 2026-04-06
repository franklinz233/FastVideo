#!/bin/bash

source "$(dirname "$0")/env.sh"

# CONFIG="examples/train/configs/longvid_sft/finetune_wan2.1_longvid_sft_1.3B_161f.yaml"
CONFIG="examples/train/configs/longvid_sft/finetune_wan2.1_longvid_sft_1.3B_161f_yarn.yaml"

torchrun \
    --nproc_per_node=8 \
    --master_port=29501 \
    -m fastvideo.train.entrypoint.train \
    --config $CONFIG

