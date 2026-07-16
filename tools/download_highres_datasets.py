import os
import sys
import argparse
import urllib.request
import zipfile
import shutil
from PIL import Image

# Setup sys.path to bypass local "datasets" folder conflict
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_BASE = os.path.join(PROJECT_DIR, "models", "CodeFormer", "datasets", "ffhq", "ffhq_512")

# Setup clean path for HuggingFace datasets
sys.path = [p for p in sys.path if p not in ('', '.')]
from datasets import load_dataset

def download_div2k(test_run=False):
    print("\n" + "="*50)
    print("  DOWNLOADING DIV2K DATASET (2K RESOLUTION)")
    print("="*50)
    
    target_dir = os.path.join(DATASET_BASE, "div2k")
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if we already have images
    existing = len([f for f in os.listdir(target_dir) if f.endswith(".png")])
    if existing >= (5 if test_run else 800):
        print(f"[OK] DIV2K already contains {existing} images. Skipping download.")
        return
        
    zip_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    zip_path = os.path.join(PROJECT_DIR, "DIV2K_train_HR.zip")
    extract_dir = os.path.join(PROJECT_DIR, "DIV2K_temp")
    
    # In test mode we don't download 3.5GB zip, we just download 5 images directly if possible
    if test_run:
        print("[Test Mode] Downloading 5 sample images directly from DIV2K repo...")
        for i in range(1, 6):
            img_url = f"https://raw.githubusercontent.com/eugenesiow/super-image-data/master/div2k/DIV2K_train_HR/{i:04d}.png"
            dest = os.path.join(target_dir, f"div2k_{i:04d}.png")
            try:
                urllib.request.urlretrieve(img_url, dest)
                print(f"  Saved sample {i}/5 -> {dest}")
            except Exception as e:
                print(f"  Failed sample {i}: {e}")
        return

    print(f"Downloading {zip_url} (approx. 3.5 GB)...")
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, read_so_far * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent:.2f}% ({read_so_far / 1024 / 1024:.2f} MB of {total_size / 1024 / 1024:.2f} MB)")
        else:
            sys.stdout.write(f"\rProgress: {read_so_far / 1024 / 1024:.2f} MB")
        sys.stdout.flush()
        
    try:
        urllib.request.urlretrieve(zip_url, zip_path, reporthook=report_hook)
        print("\n[OK] DIV2K Zip downloaded. Extracting...")
        
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        print("[*] Copying images to target directory...")
        copied = 0
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(".png"):
                    src = os.path.join(root, file)
                    dest = os.path.join(target_dir, f"div2k_{file}")
                    shutil.copy2(src, dest)
                    copied += 1
                    
        print(f"[OK] Successfully extracted and copied {copied} DIV2K images.")
        
    except Exception as e:
        print(f"\n[FAIL] Error occurred during DIV2K setup: {e}")
    finally:
        # Cleanup
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
        except Exception:
            pass

def download_huggingface_dataset(repo_name, target_subdir, max_images, image_key='image', prefix='img', test_run=False):
    print("\n" + "="*50)
    print(f"  STREAMING DATASET FROM HF: {repo_name}")
    print("="*50)
    
    target_dir = os.path.join(DATASET_BASE, target_subdir)
    os.makedirs(target_dir, exist_ok=True)
    
    limit = 5 if test_run else max_images
    
    existing = len([f for f in os.listdir(target_dir) if f.endswith(".png")])
    if existing >= limit:
        print(f"[OK] {target_subdir} already contains {existing} images. Skipping.")
        return
        
    print(f"Streaming from HF hub. Target count: {limit} images...")
    try:
        ds = load_dataset(repo_name, split="train", streaming=True)
        saved = 0
        for i, item in enumerate(ds):
            if saved >= limit:
                break
                
            dest = os.path.join(target_dir, f"{prefix}_{saved:06d}.png")
            if os.path.exists(dest):
                saved += 1
                continue
                
            try:
                # Retrieve the PIL image
                pil_img = item[image_key]
                if not isinstance(pil_img, Image.Image):
                    # In case it's a dict with bytes
                    import io
                    pil_img = Image.open(io.BytesIO(pil_img['bytes']))
                
                # Keep high-res, save as PNG
                pil_img.convert("RGB").save(dest, "PNG")
                saved += 1
                if saved % 100 == 0 or test_run:
                    print(f"  -> Saved {saved}/{limit} images")
            except Exception as e:
                print(f"  [Error] Failed to save image index {i}: {e}")
                
        print(f"[OK] Successfully streamed and saved {saved} images to {target_dir}")
    except Exception as e:
        print(f"[FAIL] Error streaming dataset {repo_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download and integrate high-resolution datasets for super-resolution training.")
    parser.add_argument("--test-run", action="store_true", help="Only download 5 images from each dataset for verification.")
    args = parser.parse_args()
    
    # 1. Download DIV2K (2K resolution)
    download_div2k(test_run=args.test_run)
    
    # 2. Stream Flickr2K Subset (2K resolution, yangtao9009/Flickr2K)
    download_huggingface_dataset(
        repo_name="yangtao9009/Flickr2K",
        target_subdir="flickr2k_subset",
        max_images=1000,
        image_key="image",
        prefix="flickr2k",
        test_run=args.test_run
    )
    
    # 3. Stream FFHQ-1024x1024 Subset (1024px resolution, Iceclear/FFHQ-HQ1024)
    download_huggingface_dataset(
        repo_name="Iceclear/FFHQ-HQ1024",
        target_subdir="ffhq1024_subset",
        max_images=10000,
        image_key="image",
        prefix="ffhq1024",
        test_run=args.test_run
    )

if __name__ == "__main__":
    main()
