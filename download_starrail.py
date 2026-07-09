import os
import sys
import io
import zipfile
import shutil
import requests
import time
from PIL import Image

# ─────────────────────────────────────────────────────────────
# 1. CyberHarem Cropped Portrait Datasets (23 Characters)
# ─────────────────────────────────────────────────────────────
CYBERHAREM_CHARS = [
    "march_7th_starrail",
    "silver_wolf_starrail",
    "bronya_starrail",
    "herta_starrail",
    "seele_starrail",
    "clara_starrail",
    "tingyun_starrail",
    "himeko_starrail",
    "asta_starrail",
    "qingque_starrail",
    "bailu_starrail",
    "serval_starrail",
    "natasha_starrail",
    "sushang_starrail",
    "yukong_starrail",
    "pela_starrail",
    "kafka_starrail",
    "fu_xuan_starrail",
    "jingliu_starrail",
    "huohuo_starrail",
    "guinaifen_starrail",
    "hanya_starrail",
    "hook_starrail"
]

IMAGES_PER_CHARACTER = 30  # 30 cropped portraits per character

# ─────────────────────────────────────────────────────────────
# 2. Official High-Resolution Skins (44 Characters)
# ─────────────────────────────────────────────────────────────
SKINS_API_ROOT = "https://huggingface.co/api/datasets/deepghs/game_character_skins/tree/main/starrail"
SKINS_RESOLVE_ROOT = "https://huggingface.co/datasets/deepghs/game_character_skins/resolve/main"

# ─────────────────────────────────────────────────────────────
# 3. Safebooru Crawler (For New Releases & All Characters)
# ─────────────────────────────────────────────────────────────
NEW_CHARACTERS_TAGS = {
    "acheron": "acheron_(honkai:_star_rail)",
    "firefly": "firefly_(honkai:_star_rail)",
    "robin": "robin_(honkai:_star_rail)",
    "sparkle": "sparkle_(honkai:_star_rail)",
    "sunday": "sunday_(honkai:_star_rail)",
    "aventurine": "aventurine_(honkai:_star_rail)",
    "boothill": "boothill_(honkai:_star_rail)",
    "feixiao": "feixiao_(honkai:_star_rail)",
    "yunli": "yunli_(honkai:_star_rail)",
    "lingsha": "lingsha_(honkai:_star_rail)",
    "rappa": "rappa_(honkai:_star_rail)",
    "jiaoqiu": "jiaoqiu_(honkai:_star_rail)",
    "cerydra": "cerydra_(honkai:_star_rail)",
    "hysilens": "hysilens_(honkai:_star_rail)",
    "general_hsr": "honkai:_star_rail"  # General wallpapers/illustrations
}

IMAGES_PER_TAG = 40  # Download 40 images per specific tag

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def download_and_extract_cyberharem(char_name: str, target_dir: str, temp_dir: str):
    """Download dataset-800.zip from CyberHarem and extract target number of images."""
    url = f"https://huggingface.co/datasets/CyberHarem/{char_name}/resolve/main/dataset-800.zip"
    zip_path = os.path.join(temp_dir, f"{char_name}.zip")
    extract_temp = os.path.join(temp_dir, char_name)

    print(f"\n[+] CyberHarem: {char_name}")
    
    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=30)
        if response.status_code != 200:
            url = f"https://huggingface.co/datasets/CyberHarem/{char_name}/resolve/main/dataset-raw.zip"
            response = requests.get(url, headers=HEADERS, stream=True, timeout=30)
            if response.status_code != 200:
                print(f"    [FAIL] Failed to download dataset for {char_name}")
                return 0

        # Save zip
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract zip
        os.makedirs(extract_temp, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_temp)

        # Find and process images
        extracted_images = []
        for root, _, files in os.walk(extract_temp):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    extracted_images.append(os.path.join(root, file))

        saved_count = 0
        for i, img_path in enumerate(extracted_images):
            if saved_count >= IMAGES_PER_CHARACTER:
                break
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                
                dest_filename = f"starrail_ch_{char_name}_{saved_count:05d}.png"
                dest_path = os.path.join(target_dir, dest_filename)
                
                img.save(dest_path, "PNG")
                saved_count += 1
            except Exception as e:
                pass

        print(f"    [OK] Saved {saved_count} cropped images.")
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

def download_official_skins(target_dir: str):
    """Download official skins/artworks dynamically from deepghs/game_character_skins."""
    print("\n" + "=" * 60)
    print("  DOWNLOADING OFFICIAL HIGH-RES ARTWORKS")
    print("=" * 60)

    try:
        r = requests.get(SKINS_API_ROOT, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"[FAIL] Unable to query skins tree API: {r.status_code}")
            return 0
        
        items = r.json()
        characters = [item for item in items if item["type"] == "directory"]
        print(f"Found {len(characters)} characters in Star Rail skins database.")

        total_downloaded = 0
        for idx, char_dir in enumerate(characters):
            char_path = char_dir["path"]
            char_name = os.path.basename(char_path)

            print(f"[+] Skins: {char_name} ({idx+1}/{len(characters)})")
            
            # Query files in character folder
            files_api = f"https://huggingface.co/api/datasets/deepghs/game_character_skins/tree/main/{char_path}"
            fr = requests.get(files_api, headers=HEADERS, timeout=30)
            if fr.status_code != 200:
                print(f"    [FAIL] Unable to query files API for {char_name}")
                continue

            file_items = fr.json()
            image_files = [f for f in file_items if f["type"] == "file" and f["path"].lower().endswith(('.png', '.jpg', '.jpeg'))]

            for img_idx, img_file in enumerate(image_files):
                img_path = img_file["path"]
                img_filename = os.path.basename(img_path)
                
                if img_filename.startswith('.') or "chibi" in img_filename.lower():
                    continue

                resolve_url = f"{SKINS_RESOLVE_ROOT}/{img_path}"
                try:
                    ir = requests.get(resolve_url, headers=HEADERS, timeout=30)
                    if ir.status_code == 200:
                        img = Image.open(io.BytesIO(ir.content)).convert("RGB")
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        
                        dest_filename = f"starrail_skin_{char_name}_{img_idx:03d}.png"
                        dest_path = os.path.join(target_dir, dest_filename)
                        img.save(dest_path, "PNG")
                        total_downloaded += 1
                except Exception as e:
                    pass
        print(f"[OK] Downloaded {total_downloaded} official character skins/artworks.")
        return total_downloaded
    except Exception as e:
        print(f"[FAIL] Error downloading official skins: {e}")
        return 0

def download_from_safebooru(target_dir: str):
    """Crawl images from Safebooru for new releases and general HSR tags."""
    print("\n" + "=" * 60)
    print("  CRAWLING SAFEBOORU FOR NEW CHARACTER IMAGES & WALLPAPERS")
    print("=" * 60)

    total_downloaded = 0

    for nickname, tag in NEW_CHARACTERS_TAGS.items():
        print(f"[+] Safebooru Tag: {tag}")
        saved_tag_count = 0
        page = 0
        
        while saved_tag_count < IMAGES_PER_TAG:
            # Query Safebooru DAPI
            api_url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&tags={tag}&limit=50&pid={page}"
            try:
                r = requests.get(api_url, headers=HEADERS, timeout=20)
                if r.status_code != 200:
                    print(f"    [FAIL] HTTP status {r.status_code} for tag {nickname}")
                    break
                
                try:
                    posts = r.json()
                except Exception:
                    # If empty or not json, break
                    break
                
                if not posts:
                    break

                for post in posts:
                    if saved_tag_count >= IMAGES_PER_TAG:
                        break
                    
                    file_url = post.get("file_url")
                    if not file_url:
                        continue
                    
                    # Fix escaped slashes in JSON
                    file_url = file_url.replace("\\/", "/")
                    if not file_url.startswith("http"):
                        file_url = "https:" + file_url
                    
                    try:
                        # Download image
                        ir = requests.get(file_url, headers=HEADERS, timeout=15)
                        if ir.status_code == 200:
                            img = Image.open(io.BytesIO(ir.content)).convert("RGB")
                            img = img.resize((512, 512), Image.Resampling.LANCZOS)
                            
                            dest_filename = f"starrail_sb_{nickname}_{post['id']}.png"
                            dest_path = os.path.join(target_dir, dest_filename)
                            img.save(dest_path, "PNG")
                            saved_tag_count += 1
                            total_downloaded += 1
                            
                            # Small delay to avoid hitting rate limits
                            time.sleep(0.1)
                    except Exception:
                        pass
                
                page += 1
                time.sleep(0.5)
            except Exception as e:
                print(f"    [FAIL] Connection error for {nickname}: {e}")
                break

        print(f"    [OK] Saved {saved_tag_count} images for {nickname}.")
        
    print(f"[OK] Downloaded {total_downloaded} images from Safebooru.")
    return total_downloaded

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
    print("  HONKAI STAR RAIL MASTER DATASET DOWNLOADER")
    print(f"  Target directory: {target_dir}")
    print("=" * 60)

    # 1. Download official high-res artworks (44 characters)
    official_count = download_official_skins(target_dir)

    # 2. Crawl Safebooru for new releases and general HSR wallpapers
    safebooru_count = download_from_safebooru(target_dir)

    # 3. Download CyberHarem cropped portraits (23 characters)
    print("\n" + "=" * 60)
    print("  DOWNLOADING CYBERHAREM CHARACTER PORTRAITS")
    print("=" * 60)
    
    cyber_count = 0
    for char in CYBERHAREM_CHARS:
        count = download_and_extract_cyberharem(char, target_dir, temp_dir)
        cyber_count += count

    # Cleanup temp directory completely
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    total_images = official_count + safebooru_count + cyber_count
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print(f"  Official high-res skins saved  : {official_count}")
    print(f"  Safebooru images saved         : {safebooru_count}")
    print(f"  CyberHarem portraits saved     : {cyber_count}")
    print(f"  Total HSR images in dataset    : {total_images}")
    print(f"  Dataset path: {target_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
