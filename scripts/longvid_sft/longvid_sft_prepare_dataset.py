#!/usr/bin/env python3
"""
Convert Sekai game walking dataset CSV to FastVideo training format.
"""
import json
import subprocess
import pandas as pd
from pathlib import Path

def get_video_info(video_path):
    """Get FPS, frame count, and resolution from video using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,nb_frames,width,height',
            '-of', 'json',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        stream = data['streams'][0]

        # Parse frame rate
        r_frame_rate = stream['r_frame_rate']
        num, denom = map(int, r_frame_rate.split('/'))
        fps = num / denom

        # Get frame count
        num_frames = int(stream['nb_frames'])

        # Get resolution
        width = int(stream['width'])
        height = int(stream['height'])

        return fps, num_frames, width, height
    except Exception as e:
        print(f"Error getting video info for {video_path}: {e}")
        return None, None, None, None

def main():
    csv_path = Path("/root/data_root/dataset/sekai/train/sekai-game-walking.csv")
    video_dir = Path("/root/data_root/dataset/sekai/sekai-game-walking")
    output_dir = Path("/root/data_root/dataset/sekai/sekai-game-walking-formatted")

    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path)

    # Create videos2caption.json
    videos2caption = []
    for idx, row in df.iterrows():
        video_file = row['videoFile']
        caption = row['caption']

        # Check if video exists
        video_path = video_dir / video_file
        if video_path.exists():
            # Get video metadata
            fps, num_frames, width, height = get_video_info(video_path)

            if fps is not None and num_frames is not None and width is not None and height is not None:
                videos2caption.append({
                    "path": video_file,
                    "cap": caption,
                    "fps": fps,
                    "num_frames": num_frames,
                    "resolution": {
                        "width": width,
                        "height": height
                    }
                })

                # Create symlink in output videos directory
                target_link = videos_dir / video_file
                if not target_link.exists():
                    target_link.symlink_to(video_path)

                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(df)} videos...")

    # Save videos2caption.json
    output_json = output_dir / "videos2caption.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(videos2caption, f, indent=2, ensure_ascii=False)

    print(f"\nCreated dataset with {len(videos2caption)} videos")
    print(f"Output directory: {output_dir}")
    print(f"videos2caption.json: {output_json}")

if __name__ == "__main__":
    main()
