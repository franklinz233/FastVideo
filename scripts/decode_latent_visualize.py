#!/usr/bin/env python3
"""
Decode a VAE latent from a preprocessed parquet file and save as video/gif.

Usage:
    python decode_latent_visualize.py --parquet <path> --row 0 --output /tmp/decoded.mp4
"""
import argparse
import os
import numpy as np
import pyarrow.parquet as pq
import torch
from diffusers import AutoencoderKLWan


def decode_and_save(parquet_path: str, row_idx: int, output_path: str, model_path: str):
    # Load parquet row
    table = pq.read_table(parquet_path)
    row = table.slice(row_idx, 1).to_pydict()

    print(f"File: {row['file_name'][0]}")
    print(f"Caption: {row['caption'][0][:120]}")
    print(f"Width: {row['width'][0]}, Height: {row['height'][0]}, num_frames: {row['num_frames'][0]}, fps: {row['fps'][0]}")

    # Reconstruct latent tensor
    lat_np = np.frombuffer(row['vae_latent_bytes'][0], dtype=row['vae_latent_dtype'][0]).copy()
    lat_np = lat_np.reshape(row['vae_latent_shape'][0])
    latent = torch.from_numpy(lat_np).unsqueeze(0).to(torch.float32)  # (1, C, T, H, W)
    print(f"Latent shape: {latent.shape}")

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to("cuda")
    vae.eval()

    # Scale latent (undo the scaling done during encoding)
    # Wan VAE uses scaling_factor
    scaling_factor = vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else 1.0
    print(f"VAE scaling_factor: {scaling_factor}")
    latent = latent.to("cuda") / scaling_factor

    # Decode
    print("Decoding latent...")
    with torch.no_grad():
        decoded = vae.decode(latent).sample  # (1, C, T, H, W)
    print(f"Decoded shape: {decoded.shape}")

    # Convert to uint8 video frames
    # decoded is in [-1, 1]
    frames = decoded[0].permute(1, 2, 3, 0).cpu().float()  # (T, H, W, C)
    frames = ((frames.clamp(-1, 1) + 1) / 2 * 255).byte().numpy()
    print(f"Frames: {frames.shape}, dtype: {frames.dtype}")

    # Save as video using torchvision
    import torchvision.io as tvio
    frames_tensor = torch.from_numpy(frames)  # (T, H, W, C)
    fps = row['fps'][0]
    tvio.write_video(output_path, frames_tensor, fps=fps)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Path to parquet file")
    parser.add_argument("--row", type=int, default=0, help="Row index to decode")
    parser.add_argument("--output", default="/tmp/decoded.mp4", help="Output video path")
    parser.add_argument("--model_path",
                        default="/root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b",
                        help="Path to Wan model")
    args = parser.parse_args()
    decode_and_save(args.parquet, args.row, args.output, args.model_path)
