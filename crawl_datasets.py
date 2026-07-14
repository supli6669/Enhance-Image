import os
import sys
import io
import pandas as pd
from PIL import Image
import requests
import time
from bs4 import BeautifulSoup
# Selenium for dynamic page rendering (Pinterest loads images via JS)
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ─────────────────────────────────────────────
# HuggingFace dataset sources
# ─────────────────────────────────────────────
SOURCES = {
    "face": {
        "parquet_url": "https://huggingface.co/datasets/Ryan-sjtu/ffhq512-caption/resolve/main/data/train-00000-of-00054-9b5f7c3e6bc03b3b.parquet",
        "num_images": 300,
        "subdir": "faces",
        "prefix": "face",
        "description": "FFHQ 512x512 real face photos",
    },
    "landscape": {
        "parquet_url": "https://huggingface.co/datasets/mertcobanov/nature-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "num_images": 300,
        "subdir": "landscapes",
        "prefix": "landscape",
        "description": "Nature & landscape scenery (50k images)",
    },
    "anime": {
        "parquet_url": "https://huggingface.co/datasets/amirali900/Anime-Face-Dataset-10k/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "num_images": 300,
        "subdir": "anime",
        "prefix": "anime",
        "description": "Anime Face Dataset 10k illustrations",
    },
    "blue_archive": {
        # This source is a web‑scrape target rather than a parquet dataset.
        "type": "web",
        "search_url": "https://www.pinterest.com/search/pins/?q=blue%20archive",
        "num_images": 300,
        "subdir": "blue_archive",
        "prefix": "blue",
        "description": "Blue Archive images scraped from Pinterest",
    },
}

# Fallback: direct image download lists for anime / landscape
ANIME_FALLBACK_URLS = [
    "https://huggingface.co/datasets/huggan/anime-faces/resolve/main/data/train-00000-of-00001.parquet",
]

LANDSCAPE_FALLBACK_URLS = [
    "https://huggingface.co/datasets/jlbaker361/flickr_humans/resolve/main/data/train-00000-of-00001.parquet",
]


def load_parquet_safe(url: str) -> pd.DataFrame | None:
    """Try to load a parquet file from URL, return None on failure."""
    print(f"  Loading: {url}")
    try:
        df = pd.read_parquet(url)
        print(f"  [OK] Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return None


def find_image_column(df: pd.DataFrame) -> str | None:
    """Detect which column holds image data (bytes dict or PIL-compatible)."""
    for col in df.columns:
        sample = df[col].iloc[0]
        if isinstance(sample, dict) and "bytes" in sample:
            return col
        if isinstance(sample, bytes):
            return col
    return None


def extract_image_bytes(cell) -> bytes | None:
    """Extract raw image bytes from a dataframe cell regardless of format."""
    if isinstance(cell, dict):
        return cell.get("bytes")
    if isinstance(cell, bytes):
        return cell
    return None


# ---------------------------------------------------------------------
# Helper functions for web‑scraping sources
# ---------------------------------------------------------------------
def scrape_image_urls(search_url: str, max_images: int) -> list:
    """Scrape image URLs from a Pinterest search page.

    The function fetches the HTML of the search results, extracts ``<img>``
    tags and returns up to ``max_images`` URLs that end with common image file
    extensions. Pagination is handled by appending ``&page=N`` to the query URL
    when needed.
    """
    print(f"  Scraping Pinterest for up to {max_images} images …")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    collected: list[str] = []
    page = 1
    # Initialize a headless Chrome driver for dynamic content.
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    while len(collected) < max_images:
        # Build the URL for the current page.
        url = (
            f"{search_url}&page={page}" if "page=" not in search_url else f"{search_url.split('page=')[0]}page={page}"
        )
        try:
            driver.get(url)
            # Give the page a moment to load dynamic content.
            driver.implicitly_wait(3)
            page_html = driver.page_source
        except Exception as e:
            print(f"  [WARN] Selenium failed to load page {page}: {e}")
            break

        # Parse the rendered HTML.
        soup = BeautifulSoup(page_html, "html.parser")
        imgs = soup.find_all("img")
        new_urls = [img.get("src") for img in imgs if img.get("src")]
        # Filter duplicates and keep only image files.
        new_urls = [u for u in new_urls if u not in collected and u.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not new_urls:
            # No more new images on this page; stop pagination.
            break
        collected.extend(new_urls)
        print(f"  Found {len(collected)} image URLs so far …")
        page += 1

    # Clean up the driver.
    driver.quit()
    return collected[:max_images]


def download_images_from_urls(urls: list, save_dir: str, prefix: str, start_idx: int, target: int) -> int:
    """Download, resize and save images from a list of URLs.

    Mirrors the parquet pipeline: images are resized to 512×512 and saved as PNG
    files named ``{prefix}_{index:05d}.png``.
    """
    saved = 0
    for url in urls:
        if saved >= target:
            break
        try:
            resp = requests.get(url, stream=True, timeout=10)
            resp.raise_for_status()
            img = Image.open(resp.raw).convert("RGB")
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            filename = f"{prefix}_{start_idx + saved:05d}.png"
            filepath = os.path.join(save_dir, filename)
            img.save(filepath, "PNG")
            saved += 1
            if saved % 20 == 0 or saved == target:
                print(f"  [{prefix}] {saved}/{target} images saved …")
        except Exception as e:
            print(f"  [warn] Failed to download {url}: {e}")
            continue
    return saved

def crawl_source(name: str, cfg: dict, base_dataset_dir: str):
    """Download images for a single source category."""
    save_dir = os.path.join(base_dataset_dir, cfg["subdir"])
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  [{name.upper()}] {cfg['description']}")
    print(f"  Save dir : {save_dir}")
    print(f"  Target   : {cfg['num_images']} images")
    print(f"{'='*55}")

    # ---------------------------------------------------------------------
    # Determine source type. Parquet datasets are the default; "web" indicates
    # we need to scrape image URLs.
    # ---------------------------------------------------------------------
    # Compute existing highest index first – needed for both flows.
    existing = [f for f in os.listdir(save_dir) if f.endswith('.png') and f.startswith(cfg['prefix'])]
    highest_idx = -1
    for f in existing:
        try:
            idx = int(f.replace(cfg['prefix'] + '_', '').replace('.png', ''))
            highest_idx = max(highest_idx, idx)
        except ValueError:
            continue
    start_idx = highest_idx + 1

    if cfg.get('type') == 'web':
        urls = scrape_image_urls(cfg['search_url'], cfg['num_images'])
        saved = download_images_from_urls(urls, save_dir, cfg['prefix'], start_idx, cfg['num_images'])
        print(f"\n  [OK] [{name.upper()}] Done: {saved} images saved to {save_dir}")
        return saved

    # Existing parquet flow
    df = load_parquet_safe(cfg["parquet_url"])

    # Try fallbacks if primary failed
    if df is None:
        fallbacks = []
        if name == "anime":
            fallbacks = ANIME_FALLBACK_URLS
        elif name == "landscape":
            fallbacks = LANDSCAPE_FALLBACK_URLS
        for fb_url in fallbacks:
            print(f"  Trying fallback: {fb_url}")
            df = load_parquet_safe(fb_url)
            if df is not None:
                break

    if df is None:
        print(f"  [FAIL] All sources failed for '{name}'. Skipping.")
        return 0

    img_col = find_image_column(df)
    if img_col is None:
        print(f"  [FAIL] No image column detected in dataset. Columns: {list(df.columns)}")
        return 0

    print(f"  Image column: '{img_col}'")

    # Find existing highest index in this subdir to avoid overwrites
    existing = [
        f for f in os.listdir(save_dir)
        if f.endswith(".png") and f.startswith(cfg["prefix"])
    ]
    highest_idx = -1
    for f in existing:
        try:
            idx = int(f.replace(cfg["prefix"] + "_", "").replace(".png", ""))
            highest_idx = max(highest_idx, idx)
        except ValueError:
            continue

    start_idx = highest_idx + 1
    saved = 0
    target = cfg["num_images"]

    for i in range(len(df)):
        if saved >= target:
            break
        try:
            cell = df[img_col].iloc[i]
            img_bytes = extract_image_bytes(cell)
            if not img_bytes:
                continue

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((512, 512), Image.Resampling.LANCZOS)

            filename = f"{cfg['prefix']}_{start_idx + saved:05d}.png"
            filepath = os.path.join(save_dir, filename)
            img.save(filepath, "PNG")
            saved += 1

            if saved % 20 == 0 or saved == target:
                print(f"  [{name}] {saved}/{target} images saved...")

        except Exception as e:
            print(f"  [warn] Row {i} error: {e}")
            continue

    print(f"\n  [OK] [{name.upper()}] Done: {saved} images saved to {save_dir}")
    return saved


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")
    base_dataset_dir = os.path.join(codeformer_dir, "datasets", "ffhq", "ffhq_512")
    os.makedirs(base_dataset_dir, exist_ok=True)

    print("=" * 55)
    print("  MULTI-CATEGORY DATASET CRAWLER")
    print("  Categories: Face | Landscape | Anime Girl")
    print("=" * 55)

    total_saved = 0
    for name, cfg in SOURCES.items():
        saved = crawl_source(name, cfg, base_dataset_dir)
        total_saved += saved
        time.sleep(0.5)

    # Count all images recursively
    all_images = []
    for root, _, files in os.walk(base_dataset_dir):
        for f in files:
            if f.lower().endswith(".png"):
                all_images.append(f)

    print("\n" + "=" * 55)
    print(f"  CRAWL COMPLETE")
    print(f"  New images this run : {total_saved}")
    print(f"  Total dataset size  : {len(all_images)} images")
    print(f"  Dataset location    : {base_dataset_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
