"""
Kaggle Cloud GPU Automated Runner
Automates pushing training notebooks to Kaggle GPU, monitoring execution, and downloading trained model checkpoints.
"""

import os
import sys
import json
import base64
import argparse
import requests
from pathlib import Path
from datetime import datetime, timezone
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.artifact_download import destination, download
from tools.model_artifacts import inventory

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "suplo6669")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "")
BASE_URL = "https://www.kaggle.com/api/v1"
KERNEL_SLUG = "custom-ai-enhancer-stage3-training"
KERNEL_ID = 132851372

def get_auth():
    if not KAGGLE_KEY:
        raise RuntimeError("Set KAGGLE_USERNAME and KAGGLE_KEY in the environment; never store credentials in source.")
    return (KAGGLE_USERNAME, KAGGLE_KEY)

def push_training_kernel(kernel_slug=KERNEL_SLUG, model="codeformer", dataset_sources=None):
    if model == "codeformer" and not dataset_sources:
        raise ValueError("Attach a private real training dataset with --dataset-source owner/slug")
    print(f"[KaggleRunner] Preparing to push kernel '{kernel_slug}' ({model.upper()}) to Kaggle GPU...")
    
    nb_file = "train_realesrgan_kaggle.ipynb" if model.lower() == "realesrgan" else "train_kaggle.ipynb"
    nb_path = Path(__file__).resolve().parent.parent / nb_file
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
        "isPrivate": True,
        "enableGpu": True,
        "enableTpu": False,
        "enableInternet": True,
        "datasetDataSources": dataset_sources or [],
        "competitionDataSources": [],
        "kernelDataSources": [f"{KAGGLE_USERNAME}/{kernel_slug}"],
        "modelDataSources": []
    }

    url = f"{BASE_URL}/kernels/push"
    headers = {"Content-Type": "application/json"}
    
    print(f"[KaggleRunner] Uploading {nb_file} and initiating GPU execution...")
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
            raw_logs = data.get("logNullable") or data.get("logs") or ""
            if isinstance(raw_logs, str):
                try:
                    logs = json.loads(raw_logs)
                    if isinstance(logs, list):
                        print("".join([entry.get("data", "") for entry in logs]))
                    else:
                        print(raw_logs)
                except Exception:
                    print(raw_logs)
            elif isinstance(raw_logs, list):
                print("".join([entry.get("data", "") if isinstance(entry, dict) else str(entry) for entry in raw_logs]))
            else:
                print(str(raw_logs))
        except Exception as e:
            print(f"[KaggleRunner] Error parsing logs: {e}\n{resp.text[:500]}")
    else:
        print(f"Error: {resp.status_code} - {resp.text}")

def download_outputs(kernel_slug=KERNEL_SLUG, output_dir=None):
    # Each import has its own staging directory; the app never auto-loads it.
    root = Path(output_dir or Path(__file__).resolve().parents[1] / 'artifacts' /
                (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]))
    params = {"userName": KAGGLE_USERNAME, "kernelSlug": kernel_slug}
    count, seen_tokens = 0, set()
    while True:
        response = requests.get(f"{BASE_URL}/kernels/output", auth=get_auth(), params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f'Kaggle output listing HTTP {response.status_code}')
        data = response.json()
        for item in data.get('files', []):
            name = item.get('fileName', '')
            if not name.lower().endswith(('.pth', '.onnx', '.onnx.data', '.state')):
                continue
            url = item.get('url') or item.get('urlNullable')
            if not url:
                raise ValueError(f'Missing download URL for {name}')
            target = destination(root, name)
            download(url, target)
            count += 1
        token = data.get('nextPageToken')
        if not token:
            break
        if token in seen_tokens:
            raise RuntimeError('Kaggle output pagination repeated a token')
        seen_tokens.add(token)
        params['pageToken'] = token
    if not count:
        raise ValueError('No model artifacts found')
    report = {'kernel': f'{KAGGLE_USERNAME}/{kernel_slug}', 'files_downloaded': count,
              'models': inventory(root), 'promotion_status': 'not_promoted'}
    (root / 'inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Staged {count} artifacts in {root}. Review inventory and benchmark before promotion.')
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Automated Cloud GPU Training Runner")
    parser.add_argument("--push", action="store_true", help="Push notebook and start GPU training on Kaggle")
    parser.add_argument("--model", type=str, default="codeformer", choices=["codeformer", "realesrgan"], help="Target model to train")
    parser.add_argument("--status", action="store_true", help="Check current GPU execution status")
    parser.add_argument("--logs", action="store_true", help="Stream current Kaggle execution logs")
    parser.add_argument("--download", action="store_true", help="Download trained model outputs")
    parser.add_argument('--dataset-source', action='append', default=[])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    if args.push:
        push_training_kernel(model=args.model, dataset_sources=args.dataset_source)
    elif args.status:
        check_status()
    elif args.logs:
        stream_logs()
    elif args.download:
        download_outputs(output_dir=args.output_dir)
    else:
        check_status()
