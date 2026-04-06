#!/usr/bin/env python3
"""
Convert SpatialVID-HQ dataset to FastVideo training format.

Splits videos into 3 buckets based on duration at 16fps:
  81f:  5.0s <= duration < 7.5s
  161f: 7.5s <= duration < 15.0s
  321f: duration >= 15.0s
"""
import json
import os
import csv
from pathlib import Path
from multiprocessing.pool import ThreadPool

DATASET_ROOT = Path("/root/data_root/dataset/SpatialVID-HQ")
CSV_PATH = DATASET_ROOT / "data/train/SpatialVID_HQ_metadata.csv"

BUCKETS = [
    ("81f",  5.0,   7.5),
    ("161f", 7.5,  15.0),
    ("321f", 15.0, float("inf")),
]


def load_caption(row):
    ann_path = DATASET_ROOT / row["annotation path"]
    caption_file = ann_path / "caption.json"
    if not caption_file.exists():
        return None
    try:
        with open(caption_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("SceneDescription", "")
    except Exception:
        return None


def process_row(row):
    num_frames = int(row["num frames"])
    fps = float(row["fps"])
    if fps <= 0:
        return None
    duration = num_frames / fps

    bucket_name = None
    for name, lo, hi in BUCKETS:
        if lo <= duration < hi:
            bucket_name = name
            break
    if bucket_name is None:
        return None

    caption = load_caption(row)
    if not caption:
        return None

    resolution = row["resolution"]  # e.g. "1280x720"
    width, height = map(int, resolution.split("x"))

    # Strip leading "videos/" from path since the dataloader
    # prepends its own "videos/" directory
    video_rel = row["video path"]
    if video_rel.startswith("videos/"):
        video_rel = video_rel[len("videos/"):]

    return {
        "bucket": bucket_name,
        "entry": {
            "path": video_rel,
            "cap": caption,
            "fps": fps,
            "num_frames": num_frames,
            "resolution": {"width": width, "height": height},
        },
        "video_path": DATASET_ROOT / row["video path"],
    }


def main():
    print(f"Reading CSV: {CSV_PATH}")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Total rows: {len(rows)}")

    # Prepare output dirs
    output_dirs = {}
    videos_dirs = {}
    for name, _, _ in BUCKETS:
        out_dir = DATASET_ROOT / f"formatted-{name}"
        vid_dir = out_dir / "videos"
        vid_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[name] = out_dir
        videos_dirs[name] = vid_dir

    buckets_data = {name: [] for name, _, _ in BUCKETS}
    skipped = 0

    def handle_result(result, idx):
        nonlocal skipped
        if result is None:
            skipped += 1
            return
        name = result["bucket"]
        buckets_data[name].append(result["entry"])

        # Create group subdir symlink target
        video_rel = result["entry"]["path"]  # e.g. group_0001/xxx.mp4
        parts = Path(video_rel).parts  # ('group_0001', 'xxx.mp4')
        if len(parts) >= 2:
            group_dir = videos_dirs[name] / parts[0]
            group_dir.mkdir(exist_ok=True)
            link = group_dir / parts[1]
        else:
            link = videos_dirs[name] / Path(video_rel).name

        if not link.exists():
            src = result["video_path"]
            link.symlink_to(src)

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(rows)} ...")

    print("Processing rows (parallel caption loading)...")
    with ThreadPool(processes=16) as pool:
        results = pool.map(process_row, rows)

    for idx, result in enumerate(results):
        handle_result(result, idx)

    # Write videos2caption.json for each bucket
    for name, _, _ in BUCKETS:
        out_path = output_dirs[name] / "videos2caption.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(buckets_data[name], f, indent=2, ensure_ascii=False)
        print(f"Bucket {name}: {len(buckets_data[name])} videos → {out_path}")

    print(f"Skipped: {skipped} (short duration, missing caption, or error)")
    print("Done.")


if __name__ == "__main__":
    main()
