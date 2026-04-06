#!/bin/bash
# Preprocess Sekai game walking dataset for stage 3: 20s clips (321 frames @ 16 FPS)
# num_latent_t = (321 - 1) / 4 + 1 = 81

GPU_NUM=8
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATASET_PATH="/pfs_root/sc/dataset/sekai/sekai-game-walking-formatted/"
OUTPUT_DIR="/pfs_root/sc/dataset/sekai/sekai-game-walking-processed-321f/"

source "$(dirname "$0")/env.sh"

torchrun --nproc_per_node=$GPU_NUM --master_port=29501 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path $MODEL_PATH \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.video_loader_type torchvision \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR \
    --preprocess.preprocess_video_batch_size 1 \
    --preprocess.dataloader_num_workers 2 \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 321 \
    --preprocess.train_fps 16 \
    --preprocess.samples_per_file 8 \
    --preprocess.flush_frequency 8 \
    --preprocess.video_length_tolerance_range 15
