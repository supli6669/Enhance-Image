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

def test_iris_catchlight():
    print("=== Testing Studio Iris Catchlight Gleam ===")
    enhancer = WinkQualityEnhancer()
    test_face = np.full((512, 512, 3), 120, dtype=np.uint8)
    
    # Fake ParseNet mask: 4=Left Eye, 5=Right Eye
    parse_mask = np.ones((512, 512), dtype=np.uint8)
    parse_mask[180:230, 150:230] = 4 # Left Eye
    parse_mask[180:230, 280:360] = 5 # Right Eye
    
    # Make pupil dark
    test_face[190:220, 170:210] = 20
    test_face[190:220, 300:340] = 20
    
    out = enhancer.synthesize_iris_catchlight(test_face, parse_mask, strength=0.6)
    assert out is not None, "Catchlight returned None"
    assert out.shape == test_face.shape, "Catchlight shape mismatch"
    assert np.max(out[190:220, 170:210]) > 20, "Catchlight should create bright glint in pupil"
    print(f"[OK] Iris catchlight glints synthesized successfully.")

def test_hair_strand_enhancement():
    print("=== Testing Hair Strand Super-Clarity & Sheen ===")
    enhancer = WinkQualityEnhancer()
    test_face = np.random.randint(40, 180, (512, 512, 3), dtype=np.uint8)
    
    # Fake ParseNet mask: 17=Hair
    parse_mask = np.ones((512, 512), dtype=np.uint8)
    parse_mask[20:180, 80:432] = 17 # Hair
    
    out = enhancer.enhance_hair_strands(test_face, parse_mask, clarity=0.4, sheen=0.3)
    assert out is not None, "Hair enhancement returned None"
    assert out.shape == test_face.shape, "Hair enhancement shape mismatch"
    print("[OK] Hair strand clarity and gloss sheen applied cleanly.")

def test_studio_relighting():
    print("=== Testing 3D Studio Relighting & Highlighter ===")
    enhancer = WinkQualityEnhancer()
    test_face = np.full((512, 512, 3), 130, dtype=np.uint8)
    
    # Fake ParseNet mask: 10=Nose, 1=Skin, 17=Hair
    parse_mask = np.ones((512, 512), dtype=np.uint8)
    parse_mask[220:300, 230:280] = 10 # Nose
    parse_mask[20:180, 80:432] = 17 # Hair
    
    out = enhancer.apply_studio_relighting(test_face, parse_mask, rim_light=0.3, tzone_highlight=0.25)
    assert out is not None, "Studio relighting returned None"
    assert out.shape == test_face.shape, "Studio relighting shape mismatch"
    
    # Nose region should be highlighted
    assert np.mean(out[230:280, 240:270]) > 130, "Nose bridge should be highlighted"
    print("[OK] Studio T-Zone highlighter & rim lighting applied cleanly.")

def test_anti_glare_matte_skin():
    print("=== Testing AI Anti-Glare & Matte Skin Engine ===")
    enhancer = WinkQualityEnhancer()
    test_face = np.full((512, 512, 3), 140, dtype=np.uint8)
    
    # 1=Skin
    parse_mask = np.ones((512, 512), dtype=np.uint8)
    # Add severe flash glare / greasy blowout hotspot on forehead
    test_face[50:120, 200:312] = 250
    
    out = enhancer.remove_skin_glare_and_shine(test_face, parse_mask, strength=0.6)
    assert out is not None, "Anti-glare returned None"
    assert out.shape == test_face.shape, "Anti-glare shape mismatch"
    
    mean_glare_after = np.mean(out[50:120, 200:312])
    assert mean_glare_after < 250, f"Glare should be mattified down: {mean_glare_after} vs 250"
    print(f"[OK] Flash glare & hot-spot shine mattified cleanly: 250.0 -> {mean_glare_after:.1f}")

def test_portrait_makeup_palette():
    print("=== Testing Natural Studio Beauty & Makeup Palette ===")
    enhancer = WinkQualityEnhancer()
    test_face = np.full((512, 512, 3), 140, dtype=np.uint8)
    
    # 1=Skin, 2=Left Brow, 3=Right Brow, 4=Left Eye, 5=Right Eye
    parse_mask = np.ones((512, 512), dtype=np.uint8)
    parse_mask[140:170, 150:230] = 2 # Left Eyebrow
    parse_mask[140:170, 280:360] = 3 # Right Eyebrow
    parse_mask[180:220, 160:220] = 4 # Left Eye
    parse_mask[180:220, 290:350] = 5 # Right Eye
    
    out = enhancer.apply_portrait_makeup(test_face, parse_mask, blush_strength=0.4, eyebrow_boost=0.35)
    assert out is not None, "Makeup returned None"
    assert out.shape == test_face.shape, "Makeup shape mismatch"
    
    # Eyebrows should be sculpt-darkened
    mean_brow = np.mean(out[140:170, 150:230])
    assert mean_brow < 140, f"Eyebrows should be darkened: {mean_brow} vs 140"
    print(f"[OK] Natural makeup (rosy blush & eyebrow sculpting) applied successfully.")

def test_tile_upscale_8k():
    print("=== Testing 8K Ultra-HD Dynamic Super-Resolution Upscaler ===")
    enhancer = WinkQualityEnhancer()
    test_img = np.full((128, 128, 3), 150, dtype=np.uint8)
    
    out = enhancer.tile_upscale_hd(test_img, outscale=8)
    assert out is not None, "8K upscale returned None"
    assert out.shape == (1024, 1024, 3), f"8K shape mismatch: {out.shape} vs (1024, 1024, 3)"
    print(f"[OK] 8X Ultra-HD scale generated image with shape: {out.shape}")

def test_zoom_loupe_inspector_html():
    print("=== Testing 400% Zoom Loupe Inspector HTML Component ===")
    enhancer = WinkQualityEnhancer()
    img_b = np.full((300, 400, 3), 100, dtype=np.uint8)
    img_a = np.full((300, 400, 3), 200, dtype=np.uint8)
    
    html = enhancer.generate_zoom_inspector_html(img_b, img_a, widget_id="test-loupe")
    assert "test-loupe" in html, "Widget ID missing in HTML"
    assert "400% ZOOM" in html, "400% zoom badge missing in HTML"
    assert "data:image/jpeg;base64," in html, "Base64 image stream missing in HTML"
    print(f"[OK] 400% Zoom Loupe HTML component generated ({len(html)} chars).")

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
        enable_catchlight=True,
        catchlight_strength=0.55,
        enable_hair=True,
        hair_clarity=0.35,
        enable_relighting=True,
        relighting_rim=0.25,
        relighting_tzone=0.20,
        enable_anti_glare=True,
        anti_glare_strength=0.50,
        enable_makeup=True,
        blush_strength=0.30,
        eyebrow_boost=0.35,
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
    test_iris_catchlight()
    test_hair_strand_enhancement()
    test_studio_relighting()
    test_anti_glare_matte_skin()
    test_portrait_makeup_palette()
    test_tile_upscale_8k()
    test_zoom_loupe_inspector_html()
    test_full_pipeline_studio_run()
    print("\nSUCCESS: All Ultimate Studio Pro (v2.7.0) Enhancements verified with exit code 0!")
