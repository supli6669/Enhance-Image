"""Automated Kaggle GPU Verification Runner (Phase P3).

Handles:
1. Resumable chunked upload of kaggle_verify_bundle_v1 (payload.zip + manifest.json)
2. Creating and verifying private dataset `suplo6669/kaggle-verify-bundle-v1`
3. Pushing `gpu_verify.ipynb` as GPU-enabled kernel `custom-ai-enhancer-gpu-verify`
4. Polling execution status and streaming logs
5. Downloading and validating `gpu_verify_report.json`
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Optional, Tuple
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

KAGGLE_REST_URL = "https://www.kaggle.com/api/v1"
KAGGLE_BLOB_URL = "https://api.kaggle.com/v1"
BUNDLE_DIR = ROOT / "artifacts" / "kaggle_verify_bundle_v1"
DATASET_SLUG = "kaggle-verify-bundle-v1"
DATASET_TITLE = "kaggle-verify-bundle-v1"
KERNEL_SLUG = "custom-ai-enhancer-gpu-verify"
KERNEL_ID = 133718907
CHUNK_SIZE = 16 * 1024 * 1024  # 16 MB chunks for resumable GCS upload


def get_credentials() -> Tuple[str, str]:
    """Retrieve Kaggle username and key from environment or ~/.kaggle/kaggle.json."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key

    cfg_path = Path.home() / ".kaggle" / "kaggle.json"
    if cfg_path.is_file():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            username = data.get("username")
            key = data.get("key")
            if username and key:
                return username, key
        except Exception as e:
            raise RuntimeError(f"Error parsing {cfg_path}: {e}")

    raise RuntimeError("Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY or configure ~/.kaggle/kaggle.json.")


def start_blob_upload(file_name: str, file_size: int, auth: Tuple[str, str]) -> Tuple[str, str]:
    """Initiate blob upload session on Kaggle backend to obtain upload token and createUrl."""
    url = f"{KAGGLE_BLOB_URL}/blobs.BlobApiService/StartBlobUpload"
    payload = {
        "type": "DATASET",
        "name": file_name,
        "contentLength": file_size,
        "lastModifiedEpochSeconds": int(time.time()),
    }
    resp = requests.post(url, auth=auth, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"StartBlobUpload failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    return data["token"], data["createUrl"]


def upload_file_resumable(file_path: Path, upload_url: str, chunk_size: int = CHUNK_SIZE) -> None:
    """Upload a file to Google Cloud Storage via resumable PUT requests with progress updates."""
    total_bytes = file_path.stat().st_size
    file_name = file_path.name
    print(f"[Upload] Starting {file_name} ({total_bytes / (1024*1024):.1f} MB)...")

    # Query current upload progress from GCS in case of resume
    uploaded_bytes = 0
    headers = {
        "Content-Length": "0",
        "Content-Range": f"bytes */{total_bytes}",
    }
    check_resp = requests.put(upload_url, headers=headers, timeout=30)
    if check_resp.status_code == 308:
        rng = check_resp.headers.get("Range")
        if rng:
            uploaded_bytes = int(rng.split("-")[-1]) + 1
            print(f"[Upload] Resuming {file_name} from byte {uploaded_bytes} ({uploaded_bytes / (1024*1024):.1f} MB)")
    elif check_resp.status_code in (200, 201):
        print(f"[Upload] {file_name} is already completely uploaded.")
        return

    start_time = time.time()
    last_print = 0.0

    with file_path.open("rb") as f:
        f.seek(uploaded_bytes)
        while uploaded_bytes < total_bytes:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunk_len = len(chunk)
            end_byte = uploaded_bytes + chunk_len - 1
            put_headers = {
                "Content-Length": str(chunk_len),
                "Content-Range": f"bytes {uploaded_bytes}-{end_byte}/{total_bytes}",
            }
            # Upload chunk with retry
            for attempt in range(5):
                try:
                    resp = requests.put(upload_url, headers=put_headers, data=chunk, timeout=120)
                    if resp.status_code in (200, 201, 308):
                        break
                    time.sleep(2 ** attempt)
                except Exception as e:
                    if attempt == 4:
                        raise RuntimeError(f"Failed to upload chunk {uploaded_bytes}-{end_byte}: {e}")
                    time.sleep(2 ** attempt)

            uploaded_bytes += chunk_len
            now = time.time()
            if now - last_print > 3.0 or uploaded_bytes >= total_bytes:
                last_print = now
                elapsed = max(now - start_time, 0.1)
                speed = (uploaded_bytes - (f.tell() - uploaded_bytes if uploaded_bytes < total_bytes else 0)) / elapsed
                pct = (uploaded_bytes / total_bytes) * 100.0
                eta_s = (total_bytes - uploaded_bytes) / max(speed, 1.0)
                print(f"[Upload] {file_name}: {pct:.1f}% ({uploaded_bytes / (1024*1024):.1f}/{total_bytes / (1024*1024):.1f} MB) | "
                      f"ETA: {int(eta_s // 60)}m{int(eta_s % 60)}s", flush=True)

    print(f"[Upload] Completed {file_name} successfully in {time.time() - start_time:.1f}s.")


def ensure_dataset_ready(auth: Tuple[str, str]) -> str:
    """Ensure dataset `suplo6669/kaggle-verify-bundle-v1` exists and is ready on Kaggle."""
    username, _ = auth
    dataset_ref = f"{username}/{DATASET_SLUG}"
    status_url = f"{KAGGLE_REST_URL}/datasets/status/{dataset_ref}"

    resp = requests.get(status_url, auth=auth, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        status = data.get("status") if isinstance(data, dict) else str(data)
        print(f"[Dataset] Existing dataset found: '{dataset_ref}' (status: {status})")
        if status == "ready":
            return dataset_ref
        elif status == "pending":
            print("[Dataset] Waiting for existing dataset to become ready...")
            for _ in range(30):
                time.sleep(5)
                r = requests.get(status_url, auth=auth, timeout=30)
                if r.status_code == 200:
                    r_data = r.json()
                    r_st = r_data.get("status") if isinstance(r_data, dict) else str(r_data)
                    if r_st == "ready":
                        return dataset_ref
            raise RuntimeError(f"Dataset '{dataset_ref}' is stuck in pending state.")

    # Dataset does not exist -> Upload files and create it
    manifest_path = BUNDLE_DIR / "bundle_manifest.json"
    payload_path = BUNDLE_DIR / "kaggle_verify_payload.zip"

    if not manifest_path.is_file() or not payload_path.is_file():
        raise FileNotFoundError(f"Verification bundle missing files in {BUNDLE_DIR}")

    state_file = BUNDLE_DIR / ".upload_state.json"
    upload_state = {}
    if state_file.is_file():
        try:
            upload_state = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            upload_state = {}

    # 1. Upload manifest
    if "manifest_token" not in upload_state:
        print("[Dataset] Initiating blob upload for bundle_manifest.json...")
        m_token, m_url = start_blob_upload("bundle_manifest.json", manifest_path.stat().st_size, auth)
        upload_file_resumable(manifest_path, m_url)
        upload_state["manifest_token"] = m_token
        upload_state["manifest_url"] = m_url
        state_file.write_text(json.dumps(upload_state, indent=2), encoding="utf-8")

    # 2. Upload payload zip
    if "payload_token" not in upload_state:
        print("[Dataset] Initiating blob upload for kaggle_verify_payload.zip...")
        p_token, p_url = start_blob_upload("kaggle_verify_payload.zip", payload_path.stat().st_size, auth)
        upload_state["payload_token"] = p_token
        upload_state["payload_url"] = p_url
        state_file.write_text(json.dumps(upload_state, indent=2), encoding="utf-8")

    upload_file_resumable(payload_path, upload_state["payload_url"])

    # 3. Create dataset
    print(f"[Dataset] Creating private dataset '{dataset_ref}' on Kaggle...")
    create_url = f"{KAGGLE_BLOB_URL}/datasets.DatasetApiService/CreateDataset"
    create_payload = {
        "title": DATASET_TITLE,
        "slug": DATASET_SLUG,
        "ownerSlug": username,
        "licenseName": "CC0-1.0",
        "isPrivate": True,
        "files": [
            {"token": upload_state["manifest_token"]},
            {"token": upload_state["payload_token"]},
        ],
    }

    create_resp = requests.post(create_url, auth=auth, json=create_payload, timeout=60)
    if create_resp.status_code != 200:
        raise RuntimeError(f"CreateDataset failed ({create_resp.status_code}): {create_resp.text}")

    print(f"[Dataset] Dataset creation submitted: {create_resp.json()}")

    # 4. Wait for ready
    print(f"[Dataset] Waiting for '{dataset_ref}' to become ready...")
    for _ in range(60):
        time.sleep(5)
        st_resp = requests.get(status_url, auth=auth, timeout=30)
        if st_resp.status_code == 200:
            st_data = st_resp.json()
            st = st_data.get("status") if isinstance(st_data, dict) else str(st_data)
            print(f"  - Current dataset status: {st}")
            if st == "ready":
                print(f"[Dataset] SUCCESS! Dataset '{dataset_ref}' is ready.")
                if state_file.is_file():
                    state_file.unlink()  # Clean up temporary state
                return dataset_ref
            elif st == "error":
                raise RuntimeError(f"Kaggle dataset processing failed: {st_resp.text}")

    raise TimeoutError("Timed out waiting for Kaggle dataset to become ready.")


def push_verification_kernel(dataset_ref: str, auth: Tuple[str, str]) -> str:
    """Push `gpu_verify.ipynb` to Kaggle with GPU accelerator and internet enabled."""
    username, _ = auth
    nb_path = BUNDLE_DIR / "gpu_verify.ipynb"
    if not nb_path.is_file():
        raise FileNotFoundError(f"Notebook missing: {nb_path}")

    nb_text = nb_path.read_text(encoding="utf-8")
    full_slug = f"{username}/{KERNEL_SLUG}"

    payload = {
        "id": KERNEL_ID,
        "slug": full_slug,
        "newTitle": KERNEL_SLUG,
        "text": nb_text,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": True,
        "machineShape": "NvidiaTeslaT4",
        "enableTpu": False,
        "enableInternet": True,
        "datasetDataSources": [dataset_ref],
        "competitionDataSources": [],
        "kernelDataSources": [],
        "modelDataSources": [],
    }

    push_url = f"{KAGGLE_REST_URL}/kernels/push"
    print(f"[Kernel] Pushing verification notebook to '{full_slug}' (Nvidia Tesla T4 GPU)...")
    resp = requests.post(push_url, auth=auth, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Kernel push failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    if data.get("hasError"):
        raise RuntimeError(f"Kernel push rejected by Kaggle: {data.get('error')}")

    kernel_url = data.get("url") or f"https://www.kaggle.com/code/{full_slug}"
    print(f"[Kernel] SUCCESS! Kernel running: {kernel_url} (Version {data.get('versionNumber', 1)})")
    return full_slug


def monitor_and_stream_logs(kernel_slug: str, auth: Tuple[str, str]) -> str:
    """Monitor Kaggle kernel status, stream stdout logs, and wait for completion."""
    username, _ = auth
    bare_slug = kernel_slug.split("/")[-1]
    status_url = f"{KAGGLE_REST_URL}/kernels/status"
    output_url = f"{KAGGLE_REST_URL}/kernels/output"
    params = {"userName": username, "kernelSlug": bare_slug}

    seen_lines = 0
    print(f"[Monitor] Polling status for '{bare_slug}'...")

    while True:
        try:
            s_resp = requests.get(status_url, auth=auth, params=params, timeout=30)
            if s_resp.status_code == 200:
                s_data = s_resp.json()
                status = s_data.get("status", "unknown")
                msg = s_data.get("failureMessage", "")

                # Fetch logs
                o_resp = requests.get(output_url, auth=auth, params=params, timeout=30)
                if o_resp.status_code == 200:
                    o_data = o_resp.json()
                    raw_logs = o_data.get("logNullable") or o_data.get("logs") or ""
                    if isinstance(raw_logs, str):
                        lines = raw_logs.splitlines()
                    elif isinstance(raw_logs, list):
                        lines = [entry.get("data", "") if isinstance(entry, dict) else str(entry) for entry in raw_logs]
                    else:
                        lines = []

                    if len(lines) > seen_lines:
                        for line in lines[seen_lines:]:
                            safe_line = str(line).rstrip().encode("ascii", errors="replace").decode("ascii")
                            print(f"  [GPU LOG] {safe_line}", flush=True)
                        seen_lines = len(lines)

                if status in ("complete", "error", "cancelled"):
                    print(f"[Monitor] Kernel reached final status: {status.upper()}")
                    if msg:
                        print(f"[Monitor] Failure message: {msg}")
                    return status

                print(f"[Monitor] Status: {status.upper()} (waiting 15s...)", flush=True)
            else:
                print(f"[Monitor] Warning: HTTP {s_resp.status_code} while checking status")
        except Exception as e:
            print(f"[Monitor] Warning: Network check error: {e}")

        time.sleep(15)


def download_and_verify_results(kernel_slug: str, auth: Tuple[str, str]) -> bool:
    """Download output artifacts from completed kernel and verify gpu_verify_report.json."""
    username, _ = auth
    bare_slug = kernel_slug.split("/")[-1]
    output_url = f"{KAGGLE_REST_URL}/kernels/output"
    params = {"userName": username, "kernelSlug": bare_slug}

    resp = requests.get(output_url, auth=auth, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Error fetching output listing ({resp.status_code}): {resp.text}")

    data = resp.json()
    files = data.get("files", [])
    print(f"[Download] Found {len(files)} output files.")

    out_dir = BUNDLE_DIR / "verification_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = None
    # First check if gpu_verify_report.json is in files list
    target_item = next((item for item in files if item.get("fileName", "").endswith("gpu_verify_report.json")), None)
    if target_item and (target_item.get("url") or target_item.get("urlNullable")):
        furl = target_item.get("url") or target_item.get("urlNullable")
        report_path = out_dir / "gpu_verify_report.json"
        print(f"[Download] Fetching gpu_verify_report.json -> {report_path}...")
        f_resp = requests.get(furl, auth=auth, stream=True, timeout=60)
        with report_path.open("wb") as out_f:
            for chunk in f_resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    out_f.write(chunk)

    if not report_path or not report_path.is_file():
        # Check if report was printed to stdout logs
        print("[Verify] Extracting gpu_verify_report from kernel stdout logs...")
        raw_logs = data.get("logNullable") or data.get("logs") or ""
        # The logs may be a stringified JSON list of dicts with {"data": "...", "stream_name": "stdout"}
        combined_text = ""
        try:
            parsed = json.loads(raw_logs)
            if isinstance(parsed, list):
                combined_text = "".join(entry.get("data", "") for entry in parsed if isinstance(entry, dict))
            else:
                combined_text = str(raw_logs)
        except Exception:
            combined_text = str(raw_logs)

        match = re.search(r'(\{\s*"status":\s*"gpu_two_iteration_check_passed"[\s\S]*?\n\})', combined_text)
        if match:
            report_dict = json.loads(match.group(1))
            report_path = out_dir / "gpu_verify_report.json"
            report_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
            print(f"[Verify] Successfully parsed and saved: {report_path}")
        elif "gpu_two_iteration_check_passed" in combined_text:
            print("[Verify] Verified `gpu_two_iteration_check_passed` found in kernel stdout logs!")
            return True
        else:
            raise FileNotFoundError("gpu_verify_report.json not found in kernel outputs or logs.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    print("\n" + "=" * 60)
    print("KAGGLE GPU VERIFICATION REPORT:")
    print(json.dumps(report, indent=2))
    print("=" * 60 + "\n")

    if report.get("status") == "gpu_two_iteration_check_passed":
        print("[+] [GATE PASS] GPU 2-iteration verification PASSED!")
        print(f"  - Device: {report.get('gpu')}")
        print(f"  - PyTorch: {report.get('torch')}")
        print(f"  - Iterations: {report.get('iterations')}")
        print(f"  - Checkpoint readable: {report.get('checkpoint_state_readable')}")
        return True
    else:
        print(f"[-] [GATE FAIL] Verification status: {report.get('status')}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Automated Kaggle GPU Verification (Phase P3)")
    parser.add_argument("--action", choices=["full", "upload", "push", "monitor", "download"], default="full",
                        help="Action to perform: full end-to-end, or specific phase step.")
    args = parser.parse_args()

    auth = get_credentials()
    username, _ = auth
    print(f"=== Kaggle GPU Verification Runner (P3) ===")
    print(f"Authenticated as user: {username}")

    dataset_ref = f"{username}/{DATASET_SLUG}"
    kernel_slug = f"{username}/{KERNEL_SLUG}"

    if args.action in ("full", "upload"):
        dataset_ref = ensure_dataset_ready(auth)

    if args.action in ("full", "push"):
        kernel_slug = push_verification_kernel(dataset_ref, auth)

    if args.action in ("full", "monitor"):
        status = monitor_and_stream_logs(kernel_slug, auth)
        if status != "complete":
            sys.exit(1)

    if args.action in ("full", "download"):
        passed = download_and_verify_results(kernel_slug, auth)
        if not passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
