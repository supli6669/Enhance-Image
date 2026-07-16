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
# 2. Official High-Resolution Skins (HF Database)
# ─────────────────────────────────────────────────────────────
SKINS_RESOLVE_ROOT = "https://huggingface.co/datasets/deepghs/game_character_skins/resolve/main"

# ─────────────────────────────────────────────────────────────
# 3. Safebooru Crawler Configuration
# ─────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def download_and_extract_cyberharem(char_name: str, target_dir: str, temp_dir: str):
    """Download dataset-800.zip from CyberHarem and extract target number of images."""
    url = f"https://huggingface.co/datasets/CyberHarem/{char_name}/resolve/main/dataset-800.zip"
    zip_path = os.path.join(temp_dir, f"{char_name}.zip")
    extract_temp = os.path.join(temp_dir, char_name)

    print(f"\n[+] CyberHarem: {char_name}")

    # Check if we already have these images
    already_have = True
    for saved_count in range(IMAGES_PER_CHARACTER):
        dest_filename = f"starrail_ch_{char_name}_{saved_count:05d}.png"
        dest_path = os.path.join(target_dir, dest_filename)
        if not os.path.exists(dest_path):
            already_have = False
            break
    if already_have:
        print(f"    [OK] Already fully downloaded, skipping.")
        return IMAGES_PER_CHARACTER
    
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
        # Wrap cleanup in try-except to avoid WinError 32 locks on Windows
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception:
            pass
        try:
            if os.path.exists(extract_temp):
                shutil.rmtree(extract_temp)
        except Exception:
            pass

def download_official_skins(game_name: str, target_dir: str):
    """Download official skins/artworks dynamically from deepghs/game_character_skins for a specific game."""
    print("\n" + "=" * 60)
    print(f"  DOWNLOADING OFFICIAL HIGH-RES ARTWORKS: {game_name.upper()}")
    print("=" * 60)

    api_root = f"https://huggingface.co/api/datasets/deepghs/game_character_skins/tree/main/{game_name}"
    try:
        r = requests.get(api_root, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"[FAIL] Unable to query {game_name} skins tree API: {r.status_code}")
            return 0
        
        items = r.json()
        characters = [item for item in items if item["type"] == "directory"]
        print(f"Found {len(characters)} characters in {game_name} skins database.")

        total_downloaded = 0
        for idx, char_dir in enumerate(characters):
            char_path = char_dir["path"]
            char_name = os.path.basename(char_path)

            # Query files in character folder
            files_api = f"https://huggingface.co/api/datasets/deepghs/game_character_skins/tree/main/{char_path}"
            fr = requests.get(files_api, headers=HEADERS, timeout=30)
            if fr.status_code != 200:
                continue

            file_items = fr.json()
            image_files = [f for f in file_items if f["type"] == "file" and f["path"].lower().endswith(('.png', '.jpg', '.jpeg'))]

            for img_idx, img_file in enumerate(image_files):
                img_path = img_file["path"]
                img_filename = os.path.basename(img_path)
                
                if img_filename.startswith('.') or "chibi" in img_filename.lower():
                    continue

                dest_filename = f"{game_name}_skin_{char_name}_{img_idx:03d}.png"
                dest_path = os.path.join(target_dir, dest_filename)
                
                # Caching/skipping
                if os.path.exists(dest_path):
                    total_downloaded += 1
                    continue

                resolve_url = f"{SKINS_RESOLVE_ROOT}/{img_path}"
                try:
                    ir = requests.get(resolve_url, headers=HEADERS, timeout=30)
                    if ir.status_code == 200:
                        img = Image.open(io.BytesIO(ir.content)).convert("RGB")
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        img.save(dest_path, "PNG")
                        total_downloaded += 1
                except Exception:
                    pass
        print(f"[OK] Total skins checked/downloaded for {game_name}: {total_downloaded}")
        return total_downloaded
    except Exception as e:
        print(f"[FAIL] Error downloading {game_name} skins: {e}")
        return 0

def download_from_safebooru(tag: str, target_dir: str, prefix: str, max_images: int):
    """Crawl images from Safebooru for a specific tag."""
    print(f"[+] Safebooru Tag: {tag} -> Prefix: {prefix} (Max: {max_images})")
    
    saved_count = 0
    page = 0
    total_downloaded = 0
    
    while saved_count < max_images:
        api_url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&tags={tag}&limit=50&pid={page}"
        try:
            r = requests.get(api_url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                break
            
            try:
                posts = r.json()
            except Exception:
                break
            
            if not posts:
                break

            for post in posts:
                if saved_count >= max_images:
                    break
                
                file_url = post.get("file_url")
                if not file_url:
                    continue
                
                file_url = file_url.replace("\\/", "/")
                if not file_url.startswith("http"):
                    file_url = "https:" + file_url
                
                dest_filename = f"{prefix}_sb_{post['id']}.png"
                dest_path = os.path.join(target_dir, dest_filename)
                
                # Caching/skipping
                if os.path.exists(dest_path):
                    saved_count += 1
                    total_downloaded += 1
                    continue

                try:
                    ir = requests.get(file_url, headers=HEADERS, timeout=15)
                    if ir.status_code == 200:
                        img = Image.open(io.BytesIO(ir.content)).convert("RGB")
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        img.save(dest_path, "PNG")
                        saved_count += 1
                        total_downloaded += 1
                        time.sleep(0.1)
                except Exception:
                    pass
            
            page += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"    [FAIL] Connection error for tag {tag}: {e}")
            break

    print(f"    [OK] Saved {saved_count} images.")
    return total_downloaded

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")
    dataset_base = os.path.join(codeformer_dir, "datasets", "ffhq", "ffhq_512")
    
    # ─────────────────────────────────────────────────────────────
    # Paths Setup
    # ─────────────────────────────────────────────────────────────
    starrail_dir = os.path.join(dataset_base, "starrail")
    bluearchive_dir = os.path.join(dataset_base, "bluearchive")
    landscapes_dir = os.path.join(dataset_base, "landscapes")
    
    os.makedirs(starrail_dir, exist_ok=True)
    os.makedirs(bluearchive_dir, exist_ok=True)
    os.makedirs(landscapes_dir, exist_ok=True)

    temp_dir = os.path.join(project_dir, "temp_downloads")
    os.makedirs(temp_dir, exist_ok=True)

    print("=" * 60)
    print("  MULTI-DATASET MASTER CRAWLER & DOWNLOADER")
    print(f"  Workspace: {dataset_base}")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────
    # 1. HONKAI: STAR RAIL DATASET
    # ─────────────────────────────────────────────────────────────
    print("\n" + "#" * 60)
    print("  1. PROCESSING HONKAI: STAR RAIL")
    print("#" * 60)
    # A. Official Skins
    download_official_skins("starrail", starrail_dir)
    # B. CyberHarem Waifus (Commented out to avoid extremely slow zip downloads on Windows)
    # for char in CYBERHAREM_CHARS:
    #     download_and_extract_cyberharem(char, starrail_dir, temp_dir)
    
    # C. Specific New Characters & General HSR on Safebooru
    hsr_tags = {
        "acheron": ("acheron_(honkai:_star_rail)", 40),
        "firefly": ("firefly_(honkai:_star_rail)", 40),
        "robin": ("robin_(honkai:_star_rail)", 40),
        "sparkle": ("sparkle_(honkai:_star_rail)", 40),
        "sunday": ("sunday_(honkai:_star_rail)", 40),
        "aventurine": ("aventurine_(honkai:_star_rail)", 40),
        "boothill": ("boothill_(honkai:_star_rail)", 40),
        "feixiao": ("feixiao_(honkai:_star_rail)", 40),
        "yunli": ("yunli_(honkai:_star_rail)", 40),
        "lingsha": ("lingsha_(honkai:_star_rail)", 40),
        "rappa": ("rappa_(honkai:_star_rail)", 40),
        "jiaoqiu": ("jiaoqiu_(honkai:_star_rail)", 40),
        "cerydra": ("cerydra_(honkai:_star_rail)", 40),
        "hysilens": ("hysilens_(honkai:_star_rail)", 40),
        "sparxie": ("sparxie_(honkai:_star_rail)", 40),
        "yaoguang": ("yaoguang_(honkai:_star_rail)", 40),
        "castorice": ("castorice_(honkai:_star_rail)", 40),
        "evernight": ("evernight_(honkai:_star_rail)", 40),
        "hyancine": ("hyancine_(honkai:_star_rail)", 40),
        "cyrene": ("cyrene_(honkai:_star_rail)", 40),
        
        # CyberHarem waifus ported to Safebooru crawler (runs 100x faster!)
        "march_7th": ("march_7th_(honkai:_star_rail)", 30),
        "silver_wolf": ("silver_wolf_(honkai:_star_rail)", 30),
        "bronya": ("bronya_(honkai:_star_rail)", 30),
        "herta": ("herta_(honkai:_star_rail)", 30),
        "seele": ("seele_(honkai:_star_rail)", 30),
        "clara": ("clara_(honkai:_star_rail)", 30),
        "tingyun": ("tingyun_(honkai:_star_rail)", 30),
        "himeko": ("himeko_(honkai:_star_rail)", 30),
        "asta": ("asta_(honkai:_star_rail)", 30),
        "qingque": ("qingque_(honkai:_star_rail)", 30),
        "bailu": ("bailu_(honkai:_star_rail)", 30),
        "serval": ("serval_(honkai:_star_rail)", 30),
        "natasha": ("natasha_(honkai:_star_rail)", 30),
        "sushang": ("sushang_(honkai:_star_rail)", 30),
        "yukong": ("yukong_(honkai:_star_rail)", 30),
        "pela": ("pela_(honkai:_star_rail)", 30),
        "kafka": ("kafka_(honkai:_star_rail)", 30),
        "fu_xuan": ("fu_xuan_(honkai:_star_rail)", 30),
        "jingliu": ("jingliu_(honkai:_star_rail)", 30),
        "huohuo": ("huohuo_(honkai:_star_rail)", 30),
        "guinaifen": ("guinaifen_(honkai:_star_rail)", 30),
        "hanya": ("hanya_(honkai:_star_rail)", 30),
        "hook": ("hook_(honkai:_star_rail)", 30),

        "general_hsr": ("honkai:_star_rail", 350)  # Mix of all characters
    }
    for key, (tag, limit) in hsr_tags.items():
        download_from_safebooru(tag, starrail_dir, f"hsr_{key}", limit)

    # ─────────────────────────────────────────────────────────────
    # 2. BLUE ARCHIVE DATASET
    # ─────────────────────────────────────────────────────────────
    print("\n" + "#" * 60)
    print("  2. PROCESSING BLUE ARCHIVE")
    print("#" * 60)
    # A. Official Skins (113 characters)
    download_official_skins("bluearchive", bluearchive_dir)
    # B. Safebooru Characters
    ba_chars = {
        "shiroko": "shiroko_(blue_archive)",
        "aru": "aru_(blue_archive)",
        "hoshino": "hoshino_(blue_archive)",
        "nonomi": "nonomi_(blue_archive)",
        "serika": "serika_(blue_archive)",
        "momoi": "momoi_(blue_archive)",
        "midori": "midori_(blue_archive)",
        "yuzu": "yuzu_(blue_archive)",
        "mutsuki": "mutsuki_(blue_archive)",
        "iori": "iori_(blue_archive)"
    }
    for key, tag in ba_chars.items():
        download_from_safebooru(tag, bluearchive_dir, f"ba_{key}", 30)
    # C. Safebooru Mix
    download_from_safebooru("blue_archive", bluearchive_dir, "ba_general", 300)

    # ─────────────────────────────────────────────────────────────
    # 3. ANIME LANDSCAPES / BACKGROUNDS DATASET
    # ─────────────────────────────────────────────────────────────
    print("\n" + "#" * 60)
    print("  3. PROCESSING ANIME LANDSCAPES")
    print("#" * 60)
    # Use multiple popular anime scenery tags for better coverage
    landscape_tags = [
        ("scenery",             "anime_scenery",    200),
        ("no_humans",           "anime_nohuman",    100),
        ("sky",                 "anime_sky",         80),
        ("fantasy",             "anime_fantasy",     80),
        ("nature",              "anime_nature",      80),
        ("city",                "anime_city",        60),
        ("night_sky",           "anime_nightsky",    60),
        ("cherry_blossoms",     "anime_sakura",      60),
        ("ocean",               "anime_ocean",       60),
        ("forest",              "anime_forest",      60),
    ]
    for tag, prefix, max_img in landscape_tags:
        download_from_safebooru(tag, landscapes_dir, prefix, max_img)

    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("  ALL DOWNLOADS COMPLETED SUCCESSFULLY!")
    print(f"  HSR folder: {starrail_dir}")
    print(f"  Blue Archive folder: {bluearchive_dir}")
    print(f"  Anime Landscapes folder: {landscapes_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
