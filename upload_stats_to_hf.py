#!/usr/bin/env python3
"""
Upload model statistics to Hugging Face Hub.

This script uploads the data/stats directory containing model statistics 
(~13GB) to Hugging Face Hub for public access.

Requirements:
    pip install huggingface_hub

Usage:
    python upload_stats_to_hf.py

You'll need to login first:
    huggingface-cli login
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload_stats():
    """Upload data/stats directory to Hugging Face Hub."""
    
    # Configuration
    repo_id = "bkb45/ResolveUnderOverEdit-stats"
    repo_type = "dataset"
    stats_dir = Path("data/stats")
    
    # Check if stats directory exists
    if not stats_dir.exists():
        print(f"Error: {stats_dir} directory not found!")
        print("Please run this script from the repository root.")
        return
    
    print(f"📦 Preparing to upload statistics to {repo_id}...")
    print(f"📂 Source directory: {stats_dir}")
    
    # Initialize HF API
    api = HfApi()
    
    # Create the repository (if it doesn't exist)
    try:
        print(f"\n🔨 Creating repository {repo_id}...")
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=True,
            private=False
        )
        print("✅ Repository created/verified")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        print("\nMake sure you're logged in:")
        print("  huggingface-cli login")
        return
    
    # Upload the entire stats directory
    try:
        print(f"\n📤 Uploading data/stats directory...")
        print("This may take a while (~13GB of data)...")
        
        api.upload_folder(
            folder_path=str(stats_dir),
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo="data/stats",
            commit_message="Upload model statistics for EMNLP 2025 paper"
        )
        
        print("\n✅ Upload complete!")
        print(f"\n📊 View your dataset at:")
        print(f"   https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return

if __name__ == "__main__":
    print("=" * 70)
    print("Upload Model Statistics to Hugging Face Hub")
    print("=" * 70)
    upload_stats()

