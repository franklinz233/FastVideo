#!/bin/bash

source "$(dirname "$0")/env.sh"

CONFIG="examples/train/configs/longvid_sft/finetune_wan2.1_longvid_sft_1.3B_321f_from161fshift5.yaml"

torchrun \
    --nproc_per_node=8 \
    -m fastvideo.train.entrypoint.train \
    --config $CONFIG 

