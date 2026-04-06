"""
Quick check: verify all parquet files in processed-161f have correct
vae_latent_shape (dim 1 == 41 for 161-frame clips).

Usage:
    python check_parquet_latents.py [--dir PATH] [--workers N] [--expected-t 41]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def check_file(path: str, expected_t: int) -> tuple[str, list[str]]:
    """Return (path, list_of_bad_row_descriptions). Empty list = all good."""
    bad = []
    try:
        df = pd.read_parquet(path, columns=["vae_latent_shape", "id"])
        for _, row in df.iterrows():
            shape = row["vae_latent_shape"]
            t = int(shape[1])
            if t != expected_t:
                bad.append(f"  id={row['id']}  shape={list(shape)}")
    except Exception as e:
        bad.append(f"  ERROR reading file: {e}")
    return path, bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="/pfs_root/sc/dataset/SpatialVID-HQ/processed-161f/training_dataset",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--expected-t", type=int, default=41)
    args = parser.parse_args()

    root = Path(args.dir)
    files = sorted(root.rglob("*.parquet"))
    total = len(files)
    print(f"Found {total} parquet files under {root}", flush=True)

    bad_files = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(check_file, str(f), args.expected_t): f for f in files
        }
        for fut in as_completed(futures):
            path, bad = fut.result()
            done += 1
            if bad:
                bad_files.append((path, bad))
            if done % 1000 == 0 or done == total:
                print(
                    f"  [{done}/{total}] bad files so far: {len(bad_files)}",
                    flush=True,
                )

    print(f"\n=== Done ===")
    print(f"Total files checked : {total}")
    print(f"Files with bad rows : {len(bad_files)}")

    if bad_files:
        print("\nBad files:")
        for path, rows in bad_files:
            print(f"\n{path}")
            for r in rows:
                print(r)
        sys.exit(1)
    else:
        print("All OK!")


if __name__ == "__main__":
    main()
