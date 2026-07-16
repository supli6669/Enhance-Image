# datasets/resize_and_clean.py
"""
Resize all character images to 512×512 and save them into the FFHQ‑compatible folder.
Corrupted or unreadable images are skipped and reported.
Resulting folder: datasets/ffhq/ffhq_512 (flat layout).
"""

import os, cv2, glob

# Source folder containing the raw character folders
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "game_characters"))
# Destination folder expected by CodeFormer
DST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "CodeFormer", "datasets", "ffhq", "ffhq_512"))

os.makedirs(DST_ROOT, exist_ok=True)

saved = 0
failed = 0

for img_path in glob.glob(os.path.join(SRC_ROOT, "**", "*"), recursive=True):
    if not img_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        continue
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("cv2.imread returned None")
        resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        dst_path = os.path.join(DST_ROOT, os.path.basename(img_path))
        cv2.imwrite(dst_path, resized)
        saved += 1
    except Exception as e:
        failed += 1
        print(f"[BAD] {img_path} -> {e}")

print(f"[RESULT] Saved {saved} images, skipped {failed} bad files.")
