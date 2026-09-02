import os
import cv2
import sys
import numpy as np

# Add project root to sys.path
tools_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(tools_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    import tools.compat_shim
except Exception:
    pass

from pipeline import LocalAIEnhancerPipeline

def main():
    print("=" * 60)
    print("  Comprehensive Universal AI Enhancer Pipeline Test Harness")
    print("=" * 60)
    
    pipeline = LocalAIEnhancerPipeline(device='cpu')

    # -------------------------------------------------------------
    # TEST 1: Face Portrait Image Enhancement (High Likeness w=0.85)
    # -------------------------------------------------------------
    portrait_path = os.path.join(project_dir, "models", "CodeFormer", "inputs", "whole_imgs", "00.jpg")
    print(f"\n[Test 1] Testing Face Portrait Enhancement: {portrait_path}")
    assert os.path.exists(portrait_path), f"Portrait image not found at {portrait_path}"
    
    img_face = cv2.imread(portrait_path)
    assert img_face is not None, "Failed to read portrait image"
    print(f"  Input portrait shape: {img_face.shape}")
    
    res_face = pipeline.process_image(
        img_face,
        w=0.85,
        detection_model='retinaface_mobile0.25',
        upscale=2,
        preset_mode='Pure Quality',
        enable_super_clarity=True,
        clarity_strength=0.45
    )
    assert res_face is not None, "Face enhancement returned None"
    print(f"  [OK] Enhanced portrait shape: {res_face.shape}")
    cv2.imwrite(os.path.join(project_dir, "test_output_face.png"), res_face)

    # -------------------------------------------------------------
    # TEST 2: Universal Non-Face Image (Scenery/Pattern with 0 faces)
    # -------------------------------------------------------------
    print(f"\n[Test 2] Testing Universal Non-Face Enhancement (0 Faces Detected)...")
    # Synthesize a 256x256 test scenery/texture image without any human faces
    non_face_img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(non_face_img, (128, 128), 60, (0, 200, 255), -1) # Yellow sun
    cv2.rectangle(non_face_img, (0, 180), (256, 256), (50, 150, 50), -1) # Green landscape
    cv2.putText(non_face_img, "Quality AI", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    res_non_face = pipeline.process_image(
        non_face_img,
        w=0.85,
        detection_model='retinaface_mobile0.25',
        upscale=2,
        bg_upsampler='realesrgan',
        preset_mode='Pure Quality',
        enable_super_clarity=True,
        clarity_strength=0.45
    )
    assert res_non_face is not None, "Non-face enhancement returned None"
    expected_shape = (512, 512, 3)
    assert res_non_face.shape == expected_shape, f"Non-face output mismatch: got {res_non_face.shape}, expected {expected_shape}"
    print(f"  [OK] Universal Non-Face output shape: {res_non_face.shape} (2x Super-Resolution Verified)")
    cv2.imwrite(os.path.join(project_dir, "test_output_non_face.png"), res_non_face)

    # -------------------------------------------------------------
    # TEST 3: Available Model & Upscaler Discovery
    # -------------------------------------------------------------
    print(f"\n[Test 3] Verifying Multi-Model Discovery...")
    cf_models = pipeline.get_available_models()
    re_upscalers = pipeline.get_available_upscalers()
    print(f"  Discovered {len(cf_models)} CodeFormer engine(s): {list(cf_models.keys())}")
    print(f"  Discovered {len(re_upscalers)} Universal upscaler(s): {list(re_upscalers.keys())}")
    assert len(cf_models) > 0, "No CodeFormer models discovered"
    assert len(re_upscalers) > 0, "No upscalers discovered"

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED WITH 0 ERRORS! (Exit Code 0)")
    print("=" * 60)

if __name__ == "__main__":
    main()
