#!/usr/bin/env python3
"""
Script to download the SkyReels-V2 model without running inference
"""
import torch
from huggingface_hub import snapshot_download
import os

model_id = "Skywork/SkyReels-V2-DF-14B-540P"

print(f"Starting download of {model_id}...")
print("This may take 10-30 minutes depending on your internet speed.\n")

try:
    # Download the model from Hugging Face
    local_model_path = snapshot_download(
        repo_id=model_id,
        cache_dir=None,  # Uses default cache directory
    )
    print(f"\n✅ Model downloaded successfully!")
    print(f"Local path: {local_model_path}")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    exit(1)

print("\nYou can now run the video generation with:")
print(f"python generate_video_df.py --model_id {model_id} --resolution 540P --ar_step 5 --causal_block_size 5 --base_num_frames 97 --num_frames 737 --overlap_history 17 --prompt \"Your prompt here\" --addnoise_condition 20 --offload")
