#!/bin/bash
# Launch Wan 2.1 1.3B training on Sekai game walking dataset (77-frame clips)
# num_latent_t=20, 480x832, sp_size=1 (short sequence, no need for sequence parallel)

# export WANDB_MODE=disabled
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR="/root/data_root/dataset/sekai/sekai-game-walking-processed/training_dataset/"
VALIDATION_DATASET_FILE="/root/data_root/dataset/longvid_sft/validation_samples.json"
NUM_GPUS=8

# Training arguments
training_args=(
  --tracker_project_name "longvid_sft"
  --output_dir "/root/data_root/outputs/longvid_sft_wan2.1_1.3B_77f"
  --max_train_steps 5000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 20
  --num_height 480
  --num_width 832
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
)

# Parallel arguments
# sp_size=1: no sequence parallel for short clips
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim $NUM_GPUS
  --hsdp_shard_dim 1
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path $DATA_DIR
  --dataloader_num_workers 2
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file $VALIDATION_DATASET_FILE
  --validation_steps 200
  --validation_sampling_steps "50"
  --validation_guidance_scale "5.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 5e-7
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 200
  --training_state_checkpointing_steps 200
  --weight_decay 0.01
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 5
  --training_cfg_rate 0.1
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

cd /pfs_root/sc/code/FastVideo
torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
  --master_port 29501 \
    fastvideo/training/wan_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
