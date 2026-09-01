"""Unit test for Chromatic Aberration filter and Batch Processing in pipeline.py."""

from __future__ import annotations

import os
import sys
import numpy as np

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from pipeline import LocalAIEnhancerPipeline
from wink_enhancer import WinkQualityEnhancer


def test_chromatic_aberration_filter():
    print("=== Testing Chromatic Aberration Correction Filter ===")
    wink = WinkQualityEnhancer()
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    corrected = wink.correct_chromatic_aberration(img, strength=1.0)
    assert corrected.shape == img.shape
    assert corrected.dtype == np.uint8
    print("[OK] Chromatic aberration filter executed cleanly.")


def test_batch_processing_and_html_report():
    print("=== Testing Batch Processing & HTML Report Generator ===")
    pipeline = LocalAIEnhancerPipeline(device="cpu")
    
    img1 = np.full((128, 128, 3), 100, dtype=np.uint8)
    img2 = np.full((128, 128, 3), 150, dtype=np.uint8)
    
    images_dict = {
        "test_p1.png": img1,
        "test_p2.png": img2
    }

    progresses = []
    def progress_cb(stage, pct, msg):
        progresses.append((stage, pct, msg))

    batch_data = pipeline.process_batch_images(
        images_dict,
        w=0.5,
        detection_model="retinaface_mobile0.25",
        upscale=1,
        chromatic_aberration=True,
        progress_callback=progress_cb
    )

    assert batch_data["total_count"] == 2
    assert "test_p1.png" in batch_data["items"]
    assert len(progresses) >= 2
    print(f"[OK] Batch processed {batch_data['total_count']} items in {batch_data['total_duration']:.2f}s.")

    html = pipeline.generate_html_report(batch_data)
    assert "<!DOCTYPE html>" in html
    assert "test_p1.png" in html
def test_studio_beauty_engine():
    print("=== Testing Studio Beauty Engine (Teeth, Eye Glow, Tone Balance) ===")
    wink = WinkQualityEnhancer()
    face = np.full((512, 512, 3), 160, dtype=np.uint8)
    mock_mask = np.zeros((512, 512), dtype=np.uint8)
    mock_mask[200:250, 150:200] = 4 # Eye
    mock_mask[350:380, 220:280] = 11 # Teeth

    # Test teeth whitening
    whitened = wink.whiten_teeth(face, mock_mask, strength=0.5)
    assert whitened.shape == (512, 512, 3)
    assert whitened.dtype == np.uint8

    # Test ocular catchlight
    bright_eyes = wink.brighten_eyes_and_sclera(face, mock_mask, strength=0.5)
    assert bright_eyes.shape == (512, 512, 3)
    assert bright_eyes.dtype == np.uint8

    # Test lighting and skin glow tone balance
    balanced = wink.balance_portrait_lighting_and_tone(face, strength=0.3)
    assert balanced.shape == (512, 512, 3)
    assert balanced.dtype == np.uint8

    # Test full enhance_face with beauty options
    enhanced = wink.enhance_face(
        face,
        cropped_original=face,
        parse_mask=mock_mask,
        enable_teeth=True,
        enable_tone_glow=True,
        enable_eyes=True,
        enable_lips=True,
        enable_skin=True
    )
    assert enhanced.shape == (512, 512, 3)
    assert enhanced.dtype == np.uint8
    print("[OK] Studio Beauty Engine (Teeth, Eye, Glow, Tone) passed all tests!")


def main():
    test_chromatic_aberration_filter()
    test_studio_beauty_engine()
    test_batch_processing_and_html_report()
    print("SUCCESS: All batch processing, beauty suite & chromatic tests passed cleanly with exit code 0!")


if __name__ == "__main__":
    main()
