import os
import sys
import io
import pandas as pd
from PIL import Image
import requests
import time

SHARDS = [
    "train-00001-of-00054-c76023902046cea3.parquet",
    "train-00002-of-00054-b43f6d454561b047.parquet",
    "train-00003-of-00054-0637d6a1c5f28946.parquet",
    "train-00004-of-00054-17a50ec36be02fa9.parquet",
    "train-00005-of-00054-b370d8ca7905b127.parquet",
    "train-00006-of-00054-e7f70526908ad428.parquet",
    "train-00007-of-00054-36469d3079484e03.parquet",
    "train-00008-of-00054-564af51540d658a8.parquet",
    "train-00009-of-00054-57224ef56d82da23.parquet"
]

def load_parquet_safe(url):
    print(f"  Loading: {url}")
    try:
        df = pd.read_parquet(url)
        print(f"  [OK] Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return None

def find_image_column(df):
    for col in df.columns:
        sample = df[col].iloc[0]
        if isinstance(sample, dict) and "bytes" in sample:
            return col
        if isinstance(sample, bytes):
            return col
    return None

def extract_image_bytes(cell):
    if isinstance(cell, dict):
        return cell.get("bytes")
    if isinstance(cell, bytes):
        return cell
    return None

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")
    save_dir = os.path.join(codeformer_dir, "datasets", "ffhq", "ffhq_512", "faces")
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 55)
    print("  EXTENDED FFHQ REAL FACE DOWNLOADER")
    print(f"  Target directory: {save_dir}")
    print("=" * 55)
    
    # Check current count in faces subdirectory to determine start_idx
    existing = [f for f in os.listdir(save_dir) if f.endswith('.png') and f.startswith('face')]
    highest_idx = -1
    for f in existing:
        try:
            idx = int(f.replace('face_', '').replace('.png', ''))
            highest_idx = max(highest_idx, idx)
        except ValueError:
            continue
    start_idx = highest_idx + 1
    print(f"Starting index for new files: face_{start_idx:05d}.png")
    
    images_per_shard = 300
    total_saved = 0
    
    for shard_idx, shard_name in enumerate(SHARDS):
        print(f"\n[{shard_idx + 1}/{len(SHARDS)}] Processing {shard_name}...")
        url = f"https://huggingface.co/datasets/Ryan-sjtu/ffhq512-caption/resolve/main/data/{shard_name}"
        
        df = load_parquet_safe(url)
        if df is None:
            continue
            
        img_col = find_image_column(df)
        if img_col is None:
            print("  No image column found in dataframe.")
            continue
            
        saved_in_shard = 0
        for i in range(len(df)):
            if saved_in_shard >= images_per_shard:
                break
            try:
                cell = df[img_col].iloc[i]
                img_bytes = extract_image_bytes(cell)
                if not img_bytes:
                    continue
                    
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                
                filename = f"face_{start_idx + total_saved:05d}.png"
                filepath = os.path.join(save_dir, filename)
                img.save(filepath, "PNG")
                
                saved_in_shard += 1
                total_saved += 1
                
                if saved_in_shard % 50 == 0 or saved_in_shard == images_per_shard:
                    print(f"  Saved {saved_in_shard}/{images_per_shard} from this shard...")
                    
            except Exception as e:
                print(f"  [warn] Row {i} error: {e}")
                continue
                
        print(f"  Shard completed. Saved {saved_in_shard} images.")
        time.sleep(0.5)
        
    print("\n" + "=" * 55)
    print("  DOWNLOAD COMPLETED")
    print(f"  Total downloaded this run: {total_saved}")
    print(f"  Total images in directory: {len(os.listdir(save_dir))}")
    print("=" * 55)

if __name__ == "__main__":
    main()
