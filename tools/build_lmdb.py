"""
Convert the Real-ESRGAN GT dataset (loose PNGs) into an LMDB database.

LMDB gives much faster, sequential reads than thousands of small PNG files
and avoids the per-file decode/stat overhead that saturates the CPU on
Windows. This is the disk-IO optimization for CPU training.

Output is written to D:/realesrgan_lmdb (the D: drive has ~100+ GB free).
The folder name MUST end with '.lmdb' because RealESRGANDataset asserts that.

Usage:
    python tools/build_lmdb.py
"""
import os
import sys
import glob
import lmdb

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "datasets", "realesrgan_gt")
LMDB_DIR = r"D:\realesrgan.lmdb"
MAP_SIZE = 12 * 1024 * 1024 * 1024  # 12 GB upper bound (dataset is ~5.3 GB)


def main():
    if not os.path.isdir(SRC_DIR):
        print(f"[FAIL] Source dir not found: {SRC_DIR}")
        print("       Run train_realesrgan.py (or crawl_datasets.py) first.")
        sys.exit(1)

    os.makedirs(LMDB_DIR, exist_ok=True)

    pngs = sorted(glob.glob(os.path.join(SRC_DIR, "*.png")))
    if not pngs:
        print(f"[FAIL] No PNG images found in {SRC_DIR}")
        sys.exit(1)

    print(f"[*] Building LMDB from {len(pngs)} images -> {LMDB_DIR}")

    # meta_info.txt inside the lmdb folder: one key (basename w/o ext) per line
    meta_lines = []
    env = lmdb.open(LMDB_DIR, map_size=MAP_SIZE, readonly=False)
    txn = env.begin(write=True)
    key_count = 0
    for img_path in pngs:
        base = os.path.basename(img_path)
        key = os.path.splitext(base)[0]  # e.g. "0001" (no extension)
        with open(img_path, "rb") as f:
            txn.put(key.encode("ascii"), f.read())
        meta_lines.append(key)
        key_count += 1
        if key_count % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
            print(f"    ... {key_count}/{len(pngs)}")

    txn.commit()
    env.close()

    with open(os.path.join(LMDB_DIR, "meta_info.txt"), "w", encoding="utf-8") as mf:
        # Write keys WITH a .png suffix so RealESRGANDataset's
        # `line.split('.')[0]` yields the bare key (no trailing newline).
        for k in meta_lines:
            mf.write(f"{k}.png\n")

    print(f"[OK] LMDB written: {key_count} images, meta_info.txt -> {LMDB_DIR}")


if __name__ == "__main__":
    main()