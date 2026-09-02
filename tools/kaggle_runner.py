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
KERNEL_SLUG = "custom-ai-enhancer-stage3-training"
KERNEL_ID = 132851372

def get_auth():
    return (KAGGLE_USERNAME, KAGGLE_KEY)

def push_training_kernel(kernel_slug=KERNEL_SLUG):
    print(f"[KaggleRunner] Preparing to push kernel '{kernel_slug}' to Kaggle GPU...")
    
    nb_path = Path(__file__).resolve().parent.parent / "train_kaggle.ipynb"
    if not nb_path.exists():
        print(f"Error: Notebook not found at {nb_path}")
        return False
        
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_text = f.read()

    payload = {
        "id": KERNEL_ID,
        "slug": kernel_slug,
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
        print(f"[KaggleRunner] SUCCESS! Kernel pushed and running on Kaggle GPU.")
        print(f"  - Kernel URL: {data.get('url', f'https://www.kaggle.com/code/{KAGGLE_USERNAME}/{kernel_slug}')}")
        print(f"  - Version: {data.get('versionNumber', 1)}")
        return True
    else:
        print(f"[KaggleRunner] Error pushing kernel: {resp.status_code} - {resp.text}")
        return False

def check_status(kernel_slug=KERNEL_SLUG):
    url = f"{BASE_URL}/kernels/status"
    params = {"userName": KAGGLE_USERNAME, "kernelSlug": kernel_slug}
    resp = requests.get(url, auth=get_auth(), params=params)
    
    if resp.status_code == 200:
        data = resp.json()
        status = data.get("status", "unknown")
        failure_msg = data.get("failureMessage", "")
        print(f"[KaggleRunner] Kernel Status: {status.upper()}")
        if failure_msg:
            print(f"  - Message: {failure_msg}")
        return status
    else:
        print(f"[KaggleRunner] Error checking status: {resp.status_code} - {resp.text}")
        return "error"

def stream_logs(kernel_slug=KERNEL_SLUG):
    url = f"{BASE_URL}/kernels/output"
    params = {"userName": KAGGLE_USERNAME, "kernelSlug": kernel_slug}
    resp = requests.get(url, auth=get_auth(), params=params)
    if resp.status_code == 200:
        try:
            data = resp.json()
            if "logNullable" in data and data["logNullable"]:
                logs = json.loads(data["logNullable"])
                print("".join([entry.get("data", "") for entry in logs]))
            else:
                print("[KaggleRunner] No logs available yet.")
        except Exception as e:
            print(f"[KaggleRunner] Raw output: {resp.text[:300]}")
    else:
        print(f"Error: {resp.status_code} - {resp.text}")

def download_outputs(kernel_slug=KERNEL_SLUG, output_dir="weights/CodeFormer"):
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
    parser.add_argument("--status", action="store_true", help="Check current GPU execution status")
    parser.add_argument("--logs", action="store_true", help="Stream current Kaggle execution logs")
    parser.add_argument("--download", action="store_true", help="Download trained model outputs")
    args = parser.parse_args()

    if args.push:
        push_training_kernel()
    elif args.status:
        check_status()
    elif args.logs:
        stream_logs()
    elif args.download:
        download_outputs()
    else:
        check_status()
