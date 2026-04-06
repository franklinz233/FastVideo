#!/bin/bash
# Step 0: Convert sekai-game-drone CSV to FastVideo merged format
# Input:  /root/data_root/dataset/sekai/sekai-game-drone/ (mp4 files)
#         /root/data_root/dataset/sekai/train/sekai-game-drone.csv
# Output: /root/data_root/dataset/sekai/sekai-game-drone-formatted/
#           videos2caption.json
#           videos/  -> symlinks or copies

set -euo pipefail
cd /pfs_root/sc/code/FastVideo

python scripts/longvid_sft/longvid_sft_prepare_dataset.py \
    --csv /root/data_root/dataset/sekai/train/sekai-game-drone.csv \
    --video_dir /root/data_root/dataset/sekai/sekai-game-drone/ \
    --output_dir /root/data_root/dataset/sekai/sekai-game-drone-formatted/

echo "Done! Drone dataset formatted."
echo "Output: /root/data_root/dataset/sekai/sekai-game-drone-formatted/"