# datasets/crawl_game_characters_bing.py
"""
Crawl ~2000 character images per game using BingImageCrawler (more reliable than Google on Windows).
Games: Zenless Zone Zero, Wuthering Waves, Honkai Impact 3, Genshin Impact.
Images are saved under:
    datasets/game_characters/<game_slug>/
"""

import os
from icrawler.builtin import BingImageCrawler

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
GAME_QUERIES = {
    "zenless_zone_zero": "Zenless Zone Zero character fullbody",
    "wuthering_waves": "Wuthering Waves character illustration",
    "honkai_impact_3": "Honkai Impact 3 character portrait",
    "genshin_impact": "Genshin Impact character artwork",
}

IMAGES_PER_GAME = 2000  # desired number of images per game

def crawl_game(slug: str, query: str, max_num: int) -> None:
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "game_characters", slug))
    os.makedirs(out_dir, exist_ok=True)
    crawler = BingImageCrawler(storage={'root_dir': out_dir})
    crawler.crawl(keyword=query, max_num=max_num, file_idx_offset='auto')
    print(f"[DONE] {slug}: {max_num} images saved to {out_dir}")

def main() -> None:
    for slug, query in GAME_QUERIES.items():
        print(f"[START] Crawling {slug} …")
        crawl_game(slug, query, IMAGES_PER_GAME)

if __name__ == "__main__":
    main()
