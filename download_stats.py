#!/usr/bin/env python3
"""
Download model statistics from Hugging Face Hub.

This script downloads the required model statistics (~13GB) for running
the model editing experiments.

Requirements:
    pip install huggingface_hub

Usage:
    python download_stats.py [--model MODEL_NAME]
    
    # Download all models (13GB)
    python download_stats.py
    
    # Download specific model only
    python download_stats.py --model llama-3-8b
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def download_stats(model_name=None):
    """
    Download model statistics from Hugging Face Hub.
    
    Args:
        model_name: Specific model to download (e.g., 'llama-3-8b', 'gpt-j-6B').
                   If None, downloads all models.
    """
    
    repo_id = "bkb45/ResolveUnderOverEdit-stats"
    local_dir = "data/stats"
    
    print("=" * 70)
    print("Download Model Statistics from Hugging Face Hub")
    print("=" * 70)
    print(f"\n📦 Repository: {repo_id}")
    print(f"📂 Local directory: {local_dir}")
    
    if model_name:
        print(f"🎯 Downloading only: {model_name}")
        allow_patterns = [f"data/stats/{model_name}/*"]
    else:
        print("🎯 Downloading all models (~13GB)")
        allow_patterns = ["data/stats/*"]
    
    try:
        print("\n⏬ Starting download...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=".",
            allow_patterns=allow_patterns,
            resume_download=True
        )
        
        print("\n✅ Download complete!")
        print(f"📂 Statistics saved to: {local_dir}")
        
        if model_name:
            model_dir = Path(local_dir) / model_name
            if model_dir.exists():
                print(f"\n📊 Files for {model_name}:")
                for f in sorted(model_dir.rglob("*.npz")):
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"   - {f.relative_to(local_dir)} ({size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the repository exists:")
        print(f"   https://huggingface.co/datasets/{repo_id}")
        return

def main():
    parser = argparse.ArgumentParser(
        description="Download model statistics from Hugging Face Hub"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt2-large", "gpt2-xl", "gpt-j-6B", "llama-2-7b", "llama-3-8b"],
        help="Download statistics for a specific model only"
    )
    
    args = parser.parse_args()
    download_stats(args.model)

if __name__ == "__main__":
    main()

