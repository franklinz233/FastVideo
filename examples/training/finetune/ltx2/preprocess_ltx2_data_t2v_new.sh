#!/bin/bash

GPU_NUM=8
MODEL_PATH="/root/data_root/model/converted_ltx23"
DATASET_PATH="/pfs_root/sc/dataset/sekai/sekai-game-walking-formatted"
OUTPUT_DIR="$DATASET_PATH"
WITH_AUDIO=false


torchrun  --nproc_per_node=$GPU_NUM \
    --master_port=29513 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path $MODEL_PATH \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.video_loader_type torchvision \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR \
    --preprocess.with_audio $WITH_AUDIO \
    --preprocess.preprocess_video_batch_size 1 \
    --preprocess.dataloader_num_workers 0 \
    --preprocess.max_height 1088  \
    --preprocess.max_width 1920 \
    --preprocess.num_frames 121 \
    --preprocess.train_fps 24 \
    --preprocess.video_length_tolerance_range 15
