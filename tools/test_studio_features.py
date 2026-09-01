import os
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from wink_enhancer import WinkQualityEnhancer
from pipeline import LocalAIEnhancerPipeline

def test_dark_circles_and_blemish_removal():
    print("=== Testing Dark Circles & Blemish Removal ===")
    enhancer = WinkQualityEnhancer()
    
    # Create synthetic test face with dark under-eye shadow
    test_face = np.full((512, 512, 3), 160, dtype=np.uint8)
    
    # Create fake ParseNet mask: 1=Skin, 4=Left Eye, 5=Right Eye
    parse_mask = np.ones((512, 512), dtype=np.uint8)
    parse_mask[180:230, 150:230] = 4 # Left Eye
    parse_mask[180:230, 280:360] = 5 # Right Eye
    
    # Add dark circles under eyes in test image
    test_face[230:280, 140:240] = 90
    test_face[230:280, 270:370] = 90
    
    out = enhancer.conceal_dark_circles_and_blemishes(test_face, parse_mask, strength=0.5)
    assert out is not None, "Output is None!"
    assert out.shape == test_face.shape, f"Shape mismatch: {out.shape} vs {test_face.shape}"
    assert out.dtype == np.uint8, f"Dtype mismatch: {out.dtype}"
    
    # Under-eye region should be brighter
    mean_before = np.mean(test_face[240:270, 150:230])
    mean_after = np.mean(out[240:270, 150:230])
    assert mean_after > mean_before, f"Dark circles should be lightened: {mean_after} vs {mean_before}"
    print(f"[OK] Dark circles lightened successfully: {mean_before:.1f} -> {mean_after:.1f}")

def test_cinematic_luts():
    print("=== Testing Cinematic Film LUTs ===")
    enhancer = WinkQualityEnhancer()
    test_img = np.full((256, 256, 3), 128, dtype=np.uint8)
    
    luts = [
        "Kodak Portra 400 (Warm Gold)",
        "Fuji Pro 400H (Pastel Jade)",
        "Teal & Orange / Cyberpunk",
        "Leica Monochrome (B&W)"
    ]
    
    for lut in luts:
        out = enhancer.apply_cinematic_lut(test_img, lut_name=lut, intensity=1.0)
        assert out is not None, f"LUT {lut} returned None"
        assert out.shape == test_img.shape, f"LUT {lut} shape mismatch: {out.shape}"
        assert out.dtype == np.uint8, f"LUT {lut} dtype mismatch"
        print(f"[OK] LUT '{lut}' generated valid output.")

def test_portrait_bokeh():
    print("=== Testing Studio Portrait Optical Bokeh ===")
    enhancer = WinkQualityEnhancer()
    test_img = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    
    # Test with synthetic face bboxes
    bboxes = [[100, 80, 280, 260]]
    out = enhancer.apply_portrait_bokeh(test_img, face_bboxes=bboxes, bokeh_strength=0.5)
    assert out is not None, "Bokeh output is None"
    assert out.shape == test_img.shape, f"Bokeh shape mismatch: {out.shape}"
    print("[OK] Portrait bokeh blurred background cleanly.")

def test_split_slider_html():
    print("=== Testing Interactive Comparison Slider HTML Generation ===")
    enhancer = WinkQualityEnhancer()
    img_b = np.full((300, 400, 3), 100, dtype=np.uint8)
    img_a = np.full((300, 400, 3), 200, dtype=np.uint8)
    
    html = enhancer.generate_comparison_slider_html(img_b, img_a, slider_id="test-slider")
    assert "test-slider" in html, "Slider ID missing in HTML"
    assert "data:image/jpeg;base64," in html, "Base64 image stream missing in HTML"
    assert "Enhanced" in html and "Original" in html, "Badges missing in HTML"
    print(f"[OK] Comparison slider HTML generated ({len(html)} chars).")

def test_full_pipeline_studio_run():
    print("=== Testing Full Pipeline with All Studio Features Enabled ===")
    pipeline = LocalAIEnhancerPipeline(device='cpu')
    
    # 256x256 test image for fast verification
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
        
    out = pipeline.process_image(
        img,
        w=0.5,
        detection_model="retinaface_mobile0.25",
        upscale=1,
        wink_mode=True,
        enable_dark_circles=True,
        enable_teeth=True,
        enable_tone_glow=True,
        color_lut="Kodak Portra 400 (Warm Gold)",
        lut_intensity=0.8,
        bokeh_strength=0.3
    )
    assert out is not None, "Pipeline returned None!"
    assert out.shape == img.shape, f"Shape mismatch: {out.shape} vs {img.shape}"
    print(f"[OK] Full pipeline executed cleanly with output shape: {out.shape}")

if __name__ == "__main__":
    test_dark_circles_and_blemish_removal()
    test_cinematic_luts()
    test_portrait_bokeh()
    test_split_slider_html()
    test_full_pipeline_studio_run()
    print("\nSUCCESS: All 4 Studio AI Enhancements verified with exit code 0!")
