import os
import sys
import cv2
import numpy as np
import time

# Ensure project root is on sys.path
tools_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(tools_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from wink_enhancer import WinkQualityEnhancer
from pipeline import LocalAIEnhancerPipeline

def test_glamour_algorithms():
    print("==================================================")
    print("   TESTING GLAMOUR & BEAUTY SUITE (v2.9.0)")
    print("==================================================")

    enhancer = WinkQualityEnhancer()
    
    # Create test synthetic face image (512x512)
    test_face = np.full((512, 512, 3), 180, dtype=np.uint8)
    # Add skin color
    test_face[:, :] = [160, 180, 220]
    
    # Create synthetic parsing mask:
    # 1: skin, 4: left eye, 5: right eye, 12: upper lip, 13: lower lip
    parse_mask = np.zeros((512, 512), dtype=np.uint8)
    parse_mask[100:450, 100:412] = 1 # Skin
    parse_mask[200:240, 180:230] = 4 # Left eye
    parse_mask[200:240, 280:330] = 5 # Right eye
    parse_mask[360:385, 220:290] = 12 # Upper lip
    parse_mask[385:410, 220:290] = 13 # Lower lip

    # 1. Test Poreless Crystal Skin Smoothing
    print("\n--- 1. Testing Poreless Crystal Skin Smoothing ---")
    t0 = time.time()
    res_skin = enhancer.apply_crystal_skin_smoothing(test_face, parse_mask=parse_mask, strength=0.45)
    t1 = time.time()
    assert res_skin is not None and res_skin.shape == test_face.shape
    print(f"[OK] Crystal skin smoothing executed in {(t1 - t0)*1000:.2f}ms with shape {res_skin.shape}")

    # 2. Test 3D Glossy Lip Plumping & Glass Specular Highlight
    print("\n--- 2. Testing 3D Glossy Lip Plumping & Glass Specular Highlight ---")
    t0 = time.time()
    res_lips = enhancer.apply_glossy_3d_lips(test_face, parse_mask=parse_mask, gloss_strength=0.40, color_vibrance=0.25)
    t1 = time.time()
    assert res_lips is not None and res_lips.shape == test_face.shape
    # Check that lower lip center has specular highlight
    lip_diff = np.mean(res_lips[385:410, 220:290].astype(np.float32) - test_face[385:410, 220:290].astype(np.float32))
    assert lip_diff > 0, "Lower lip highlight should brighten lower lip area"
    print(f"[OK] Glossy 3D lips executed in {(t1 - t0)*1000:.2f}ms (lip luminance delta: +{lip_diff:.1f})")

    # 3. Test Doll-Eye Iris Luminescence & Limbal Ring
    print("\n--- 3. Testing Doll-Eye Iris Luminescence & Limbal Ring ---")
    t0 = time.time()
    res_eye = enhancer.apply_iris_luminescence_and_limbal_ring(test_face, parse_mask=parse_mask, depth_strength=0.45)
    t1 = time.time()
    assert res_eye is not None and res_eye.shape == test_face.shape
    print(f"[OK] Doll-eye iris luminescence executed in {(t1 - t0)*1000:.2f}ms")

    # 4. Test Sun-Kissed Golden Hour Glow
    print("\n--- 4. Testing Sun-Kissed Golden Hour Glow ---")
    t0 = time.time()
    res_golden = enhancer.apply_golden_hour_glow(test_face, warm_strength=0.25, bloom_strength=0.20)
    t1 = time.time()
    assert res_golden is not None and res_golden.shape == test_face.shape
    # Warm cast should boost red channel
    assert np.mean(res_golden[:, :, 2]) > np.mean(test_face[:, :, 2]), "Golden hour should increase red channel"
    print(f"[OK] Golden hour glow executed in {(t1 - t0)*1000:.2f}ms (red warmth: {np.mean(res_golden[:, :, 2]):.1f} vs {np.mean(test_face[:, :, 2]):.1f})")

    # 5. Test Full Pipeline Integration with Glamour Features
    print("\n--- 5. Testing Full Pipeline with All Glamour Features Enabled ---")
    pipeline = LocalAIEnhancerPipeline()
    sample_img = np.full((256, 256, 3), 128, dtype=np.uint8)
    out_img = pipeline.process_image(
        sample_img,
        upscale=1,
        enable_crystal_skin=True,
        crystal_skin_strength=0.45,
        enable_glossy_lips=True,
        lip_gloss=0.40,
        lip_vibrance=0.25,
        enable_doll_eye=True,
        doll_eye_depth=0.45,
        enable_golden_hour=True,
        golden_warmth=0.25,
        golden_bloom=0.20,
        enable_super_clarity=True,
        clarity_strength=0.40
    )
    assert out_img is not None and out_img.shape == (256, 256, 3)
    print(f"[OK] Full pipeline integration verified cleanly with output shape: {out_img.shape}")

    print("\n==================================================")
    print("SUCCESS: All Glamour & Beauty (v2.9.0) features verified with exit code 0!")
    print("==================================================")

if __name__ == "__main__":
    test_glamour_algorithms()
