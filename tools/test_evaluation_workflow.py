"""Fast regression tests for benchmark manifest validation and metric evaluators."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools"))

from evaluate_restoration import load_manifest, validate_files, BenchmarkEvaluator
from wink_enhancer import WinkQualityEnhancer


def test_manifest_validation() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_dir = root / "inputs"
        reference_dir = root / "references"
        input_dir.mkdir()
        reference_dir.mkdir()
        (input_dir / "sample.jpg").write_bytes(b"test")
        (reference_dir / "sample.jpg").write_bytes(b"test")
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "input_path", "reference_path", "category", "notes"])
            writer.writeheader()
            writer.writerow({
                "id": "sample_001",
                "input_path": "inputs/sample.jpg",
                "reference_path": "references/sample.jpg",
                "category": "jpeg_compression",
                "notes": "fixture",
            })

        samples = load_manifest(manifest)
        assert len(samples) == 1
        assert validate_files(samples) == (1, 1)

        (input_dir / "sample.jpg").unlink()
        try:
            validate_files(samples)
        except ValueError as error:
            assert "Missing input" in str(error)
        else:
            raise AssertionError("Missing benchmark input must fail validation.")

    print("SUCCESS: benchmark manifest workflow validated.")


def test_evaluator_and_animation() -> None:
    evaluator = BenchmarkEvaluator(device="cpu", enable_lpips=False, enable_identity=False)
    img_a = np.full((128, 128, 3), 120, dtype=np.uint8)
    img_b = np.full((128, 128, 3), 130, dtype=np.uint8)
    
    metrics = evaluator.evaluate_pair(img_a, img_b)
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert metrics["psnr"] > 0
    print(f"SUCCESS: evaluator pair metrics validated (PSNR: {metrics['psnr']}dB, SSIM: {metrics['ssim']}).")
    
    wink = WinkQualityEnhancer()
    gif_bytes = wink.create_comparison_animation(img_a, img_b, num_frames=8, fps=4, max_dim=128)
    assert len(gif_bytes) > 100
    print(f"SUCCESS: comparison animation generated ({len(gif_bytes)} bytes).")


def main() -> None:
    test_manifest_validation()
    test_evaluator_and_animation()
    print("ALL EVALUATION WORKFLOW TESTS PASSED CLEANLY.")


if __name__ == "__main__":
    main()
