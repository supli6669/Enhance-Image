import os
import sys
import io
import zipfile
import shutil
import requests
from PIL import Image

# ─────────────────────────────────────────────
# Honkai Star Rail Character Datasets from CyberHarem
# ─────────────────────────────────────────────
CHARACTERS = [
    "march_7th_starrail",
    "silver_wolf_starrail",
    "bronya_starrail",
    "herta_starrail",
    "seele_starrail",
    "tingyun_starrail",
    "himeko_starrail",
    "kafka_starrail",
    "fu_xuan_starrail",
    "jingliu_starrail"
]

IMAGES_PER_CHARACTER = 50  # Get 50 images per character to build a balanced dataset of 500 images

def download_and_extract_character(char_name: str, target_dir: str, temp_dir: str):
    """Download dataset-800.zip from CyberHarem and extract target number of images."""
    url = f"https://huggingface.co/datasets/CyberHarem/{char_name}/resolve/main/dataset-800.zip"
    zip_path = os.path.join(temp_dir, f"{char_name}.zip")
    extract_temp = os.path.join(temp_dir, char_name)

    print(f"\n[+] Processing character: {char_name}")
    print(f"    Downloading: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code != 200:
            print(f"    [FAIL] HTTP status {response.status_code}. Trying dataset-raw.zip...")
            # Fallback to raw if 800 doesn't exist
            url = f"https://huggingface.co/datasets/CyberHarem/{char_name}/resolve/main/dataset-raw.zip"
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code != 200:
                print(f"    [FAIL] Failed to download dataset for {char_name}")
                return 0

        # Save zip
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"    [OK] Zip downloaded.")

        # Extract zip
        os.makedirs(extract_temp, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_temp)
        print(f"    [OK] Zip extracted.")

        # Find and process images
        extracted_images = []
        for root, _, files in os.walk(extract_temp):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    extracted_images.append(os.path.join(root, file))

        print(f"    Found {len(extracted_images)} images. Processing up to {IMAGES_PER_CHARACTER}...")
        
        saved_count = 0
        for i, img_path in enumerate(extracted_images):
            if saved_count >= IMAGES_PER_CHARACTER:
                break
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Naming format: starrail_charname_000xx.png
                dest_filename = f"starrail_{char_name}_{saved_count:05d}.png"
                dest_path = os.path.join(target_dir, dest_filename)
                
                img.save(dest_path, "PNG")
                saved_count += 1
            except Exception as e:
                print(f"    [warn] Error processing {img_path}: {e}")

        print(f"    [OK] Saved {saved_count} images for {char_name}.")
        return saved_count

    except Exception as e:
        print(f"    [FAIL] Exception occurred for {char_name}: {e}")
        return 0
    finally:
        # Cleanup temp zip and extracted folders
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_temp):
            shutil.rmtree(extract_temp)

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")
    
    # Target directory for Honkai Star Rail images
    target_dir = os.path.join(codeformer_dir, "datasets", "ffhq", "ffhq_512", "starrail")
    os.makedirs(target_dir, exist_ok=True)

    # Temporary directory for zip files
    temp_dir = os.path.join(project_dir, "temp_downloads")
    os.makedirs(temp_dir, exist_ok=True)

    print("=" * 60)
    print("  HONKAI STAR RAIL DATASET DOWNLOADER")
    print(f"  Target directory: {target_dir}")
    print(f"  Temporary directory: {temp_dir}")
    print("=" * 60)

    total_downloaded = 0
    for char in CHARACTERS:
        count = download_and_extract_character(char, target_dir, temp_dir)
        total_downloaded += count

    # Cleanup temp directory completely
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print(f"  Total Honkai Star Rail images saved: {total_downloaded}")
    print(f"  Dataset path: {target_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
