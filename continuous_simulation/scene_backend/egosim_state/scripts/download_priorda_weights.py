#!/usr/bin/env python3
"""
Download PriorDA model weights from Hugging Face Hub.
Usage: python scripts/download_priorda_weights.py --output_dir ../../../artifacts/models/priorda
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def download_weights(output_dir: str, model_size: str = "vitb"):
    """Download PriorDA weights to local directory.
    
    Args:
        output_dir: Directory to save the weights
        model_size: Model size (vitb, vits, vitl, vitg)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    repo_id = "Rain729/Prior-Depth-Anything"
    
    # Files to download
    files = [
        f"depth_anything_v2_{model_size}.pth",  # Frozen and conditioned model (same file)
        f"prior_depth_anything_{model_size}.pth",  # Checkpoint
    ]
    
    print(f"Downloading PriorDA weights from {repo_id}...")
    print(f"Model size: {model_size}")
    print(f"Output directory: {output_path.absolute()}")
    print()
    
    for filename in files:
        output_file = output_path / filename
        
        if output_file.exists():
            print(f"✓ {filename} already exists, skipping...")
            continue
            
        print(f"Downloading {filename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(output_path),
                local_dir_use_symlinks=False,
            )
            print(f"✓ Downloaded to {downloaded_path}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            continue
    
    print("\nDownload complete!")
    print(f"\nTo use these weights, set the following in your code:")
    print(f"  fmde_dir='{output_path.absolute()}'")
    print(f"  cmde_dir='{output_path.absolute()}'")
    print(f"  ckpt_dir='{output_path.absolute()}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PriorDA model weights")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../../artifacts/models/priorda",
        help="Directory to save the weights (default: EgoSim/artifacts/models/priorda)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="vitb",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Model size to download (default: vitb)",
    )
    
    args = parser.parse_args()
    download_weights(args.output_dir, args.model_size)
