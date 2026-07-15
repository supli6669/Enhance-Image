import os
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
GAME_TAGS = {
    "zenless_zone_zero": "zenless_zone_zero",
    "wuthering_waves": "wuthering_waves",
    "honkai_impact_3": "honkai_impact_3rd",
    "genshin_impact": "genshin_impact",
}

IMAGES_PER_GAME = 600
MAX_WORKERS = 12  # Number of concurrent threads for downloading
OUTPUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "game_characters")
)

def download_single_image(post: dict, out_dir: str, headers: dict) -> bool:
    """Download a single image. Returns True if successful, False otherwise."""
    file_url = post.get("file_url")
    post_id = post.get("id")
    if not file_url or not post_id:
        return False
        
    # Handle relative/absolute URL
    if file_url.startswith("//"):
        file_url = "https:" + file_url
    elif file_url.startswith("/"):
        file_url = "https://safebooru.org" + file_url
    elif not file_url.startswith("http"):
        file_url = "https://safebooru.org/" + file_url
        
    ext = os.path.splitext(file_url)[1]
    if not ext:
        ext = ".jpg"
        
    if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        return False
        
    dest_path = os.path.join(out_dir, f"{post_id}{ext.lower()}")
    
    # If already downloaded, skip
    if os.path.exists(dest_path):
        return True
        
    try:
        # stream=True to check Content-Length header first
        img_res = requests.get(file_url, headers=headers, timeout=10, stream=True)
        if img_res.status_code == 200:
            content_length = img_res.headers.get("Content-Length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                # Skip files larger than 5MB to optimize download speeds
                if size_mb > 5.0:
                    img_res.close()
                    return False
            
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(img_res.raw, f)
            return True
        else:
            return False
    except Exception:
        return False

def crawl_safebooru(slug: str, tag: str, max_num: int) -> None:
    """Download images from Safebooru using DAPI JSON index concurrently."""
    out_dir = os.path.join(OUTPUT_ROOT, slug)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[START] Safebooru tags search for '{tag}' -> '{slug}'")
    
    url = "https://safebooru.org/index.php"
    params = {
        "page": "dapi",
        "s": "post",
        "q": "index",
        "json": 1,
        "limit": max_num,
        "tags": tag
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        if not response.text.strip():
            print(f"[WARN] No posts found for {tag}")
            return
            
        posts = response.json()
        if not isinstance(posts, list):
            print(f"[WARN] Expected list, got {type(posts)}")
            return
            
        print(f"Found {len(posts)} posts for '{tag}'. Downloading concurrently with {MAX_WORKERS} workers...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            futures = [
                executor.submit(download_single_image, post, out_dir, headers)
                for post in posts
            ]
            
            for future in futures:
                if future.result():
                    success_count += 1
                    if success_count % 50 == 0:
                        print(f"  Downloaded {success_count} images for {slug}...")
                        
        print(f"[DONE] {slug}: {success_count} images saved to {out_dir}\n")
        
    except Exception as e:
        print(f"[ERROR] Failed to query Safebooru for {tag}: {e}")

def main() -> None:
    print(f"Output root: {OUTPUT_ROOT}\n")
    for slug, tag in GAME_TAGS.items():
        crawl_safebooru(slug, tag, IMAGES_PER_GAME)

if __name__ == "__main__":
    main()
