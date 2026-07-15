# datasets/flatten_game_characters.py
"""
Flatten the game character image hierarchy:
- Source: datasets/game_characters/<game_name>/*
- Destination: datasets/game_characters/ (all images in one folder)
- After moving, the subfolders are removed.
"""

import os, shutil, glob

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "game_characters"))

# Gather all image paths recursively
image_paths = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
    image_paths.extend(glob.glob(os.path.join(SRC_ROOT, "**", ext), recursive=True))

moved = 0
for img_path in image_paths:
    dst_path = os.path.join(SRC_ROOT, os.path.basename(img_path))
    # If source and destination are the same, skip
    if os.path.abspath(img_path) == os.path.abspath(dst_path):
        continue
    shutil.move(img_path, dst_path)
    moved += 1

print(f"[FLATTEN] Moved {moved} images to {SRC_ROOT}")

# Optionally remove empty subfolders
for root, dirs, files in os.walk(SRC_ROOT, topdown=False):
    for d in dirs:
        dir_path = os.path.join(root, d)
        # If folder is now empty, delete it
        if not os.listdir(dir_path):
            os.rmdir(dir_path)
            print(f"[CLEAN] Removed empty folder {dir_path}")
