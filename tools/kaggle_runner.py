"""
Kaggle Cloud GPU Automated Runner
Automates pushing training notebooks to Kaggle GPU, monitoring execution, and downloading trained model checkpoints.
"""

import os
import sys
import json
import time
import requests
import argparse
from pathlib import Path

KAGGLE_USERNAME = "suplo6669"
KAGGLE_KEY = "e28e97a8021e210e91d7ed7c5603d49e"
BASE_URL = "https://www.kaggle.com/api/v1"

def get_auth():
    return (KAGGLE_USERNAME, KAGGLE_KEY)

def push_training_kernel(kernel_slug="custom-ai-enhancer-stage3-training", title="Custom AI Enhancer Stage3 Training"):
    print(f"[KaggleRunner] Preparing to push kernel '{kernel_slug}' to Kaggle GPU...")
    
    nb_path = Path(__file__).resolve().parent.parent / "train_kaggle.ipynb"
    if not nb_path.exists():
        print(f"Error: Notebook not found at {nb_path}")
        return False
        
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_text = f.read()

    payload = {
        "id": 0,
        "slug": kernel_slug,
        "newTitle": title,
        "text": nb_text,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": False,
        "enableGpu": True,
        "enableTpu": False,
        "enableInternet": True,
        "datasetDataSources": [],
        "competitionDataSources": [],
        "kernelDataSources": [],
        "modelDataSources": []
    }

    url = f"{BASE_URL}/kernels/push"
    headers = {"Content-Type": "application/json"}
    
    print("[KaggleRunner] Uploading notebook and initiating GPU execution...")
    resp = requests.post(url, auth=get_auth(), headers=headers, json=payload)
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"[KaggleRunner] SUCCESS! Kernel pushed successfully.")
        print(f"  - Kernel URL: {data.get('url', f'https://www.kaggle.com/code/{KAGGLE_USERNAME}/{kernel_slug}')}")
        print(f"  - Version: {data.get('versionNumber', 1)}")
        return True
    else:
        print(f"[KaggleRunner] Error pushing kernel: {resp.status_code} - {resp.text}")
        return False

def list_my_kernels():
    url = f"{BASE_URL}/kernels/list"
    params = {"user": KAGGLE_USERNAME}
    resp = requests.get(url, auth=get_auth(), params=params)
    if resp.status_code == 200:
        kernels = resp.json()
        print(f"[KaggleRunner] User '{KAGGLE_USERNAME}' has {len(kernels)} kernels:")
        for k in kernels:
            print(f"  - {k.get('ref')}: {k.get('title')} (https://www.kaggle.com/code/{k.get('ref')})")
        return kernels
    else:
        print(f"Error: {resp.status_code} - {resp.text}")
        return []

def download_outputs(kernel_slug="custom-ai-enhancer-stage3-training", output_dir="weights/CodeFormer"):
    print(f"[KaggleRunner] Fetching outputs from '{kernel_slug}'...")
    url = f"{BASE_URL}/kernels/output"
    params = {"userName": KAGGLE_USERNAME, "kernelSlug": kernel_slug}
    resp = requests.get(url, auth=get_auth(), params=params)
    
    if resp.status_code == 200:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        zip_file = out_path / "kaggle_output.zip"
        with open(zip_file, "wb") as f:
            f.write(resp.content)
        print(f"[KaggleRunner] Downloaded outputs to {zip_file}")
        return True
    else:
        print(f"[KaggleRunner] Error downloading outputs: {resp.status_code} - {resp.text}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Automated Cloud GPU Training Runner")
    parser.add_argument("--push", action="store_true", help="Push notebook and start GPU training on Kaggle")
    parser.add_argument("--list", action="store_true", help="List all kernels in user account")
    parser.add_argument("--download", action="store_true", help="Download trained model outputs")
    args = parser.parse_args()

    if args.push:
        push_training_kernel()
    elif args.list:
        list_my_kernels()
    elif args.download:
        download_outputs()
    else:
        push_training_kernel()
