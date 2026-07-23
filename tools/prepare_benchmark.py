"""Create a deterministic, held-out synthetic restoration benchmark.

Only real-portrait folders are selected. The generated manifest and holdout
list are ignored by Git; pass the holdout list to training to avoid leakage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "models" / "CodeFormer" / "datasets" / "ffhq" / "ffhq_512"
REAL_FOLDERS = {"faces", "pinterest"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def candidates(source: Path) -> list[Path]:
    files = []
    for folder in REAL_FOLDERS:
        directory = source / folder
        if directory.is_dir():
            files.extend(p for p in directory.rglob("*") if p.suffix.lower() in EXTENSIONS)
    return sorted(files)


def degrade(image: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blur_size = int(rng.choice([3, 5, 7]))
    output = cv2.GaussianBlur(image, (blur_size, blur_size), float(rng.uniform(0.4, 2.5)))
    h, w = output.shape[:2]
    scale = float(rng.uniform(1.5, 6.0))
    small = cv2.resize(output, (max(32, int(w / scale)), max(32, int(h / scale))), interpolation=cv2.INTER_AREA)
    output = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    output = np.clip(output.astype(np.float32) + rng.normal(0, rng.uniform(2, 25), output.shape), 0, 255).astype(np.uint8)
    quality = int(rng.integers(10, 71))
    ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG degradation failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    ranked = sorted(candidates(source), key=lambda p: hashlib.sha256(f"{args.seed}:{p.relative_to(source)}".encode()).hexdigest())
    selected = ranked[:args.count]
    if len(selected) < args.count:
        raise SystemExit(f"Need {args.count} real portraits but found {len(selected)} in {source}")
    print(f"Selected {len(selected)} held-out real portraits from {len(ranked)} candidates.")
    if args.dry_run:
        return
    benchmark = ROOT / "benchmarks"
    inputs, references = benchmark / "inputs", benchmark / "references"
    inputs.mkdir(parents=True, exist_ok=True); references.mkdir(parents=True, exist_ok=True)
    manifest = benchmark / "manifest.csv"
    holdout = benchmark / "holdout_paths.txt"
    with manifest.open("w", newline="", encoding="utf-8") as csv_file, holdout.open("w", encoding="utf-8") as holdout_file:
        writer = csv.DictWriter(csv_file, fieldnames=["id", "input_path", "reference_path", "category", "notes"]); writer.writeheader()
        for index, path in enumerate(selected, 1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None: raise RuntimeError(f"Unreadable image: {path}")
            sample_id = f"portrait_{index:04d}"
            cv2.imwrite(str(inputs / f"{sample_id}.jpg"), degrade(image, args.seed + index), [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy2(path, references / f"{sample_id}{path.suffix.lower()}")
            writer.writerow({"id": sample_id, "input_path": f"inputs/{sample_id}.jpg", "reference_path": f"references/{sample_id}{path.suffix.lower()}", "category": "synthetic_real_portrait", "notes": "deterministic holdout"})
            holdout_file.write(path.relative_to(source).as_posix() + "\n")
    print(f"Created {manifest} and {holdout}. Do not train on the listed paths.")

if __name__ == "__main__": main()
