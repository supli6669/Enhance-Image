import os
import sys
import cv2

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

def test_ab_ui_pipeline():
    print("=" * 60)
    print("  Phase 7: A/B Model & Preset Pipeline Verification")
    print("=" * 60)
    
    input_path = os.path.join(project_dir, "models", "CodeFormer", "inputs", "whole_imgs", "00.jpg")
    print(f"Loading test image: {input_path}")
    img = cv2.imread(input_path)
    assert img is not None, "Failed to load test image!"
    print(f"Input image loaded successfully, shape: {img.shape}")
    
    # 1. Initialize Pipeline
    pipeline = LocalAIEnhancerPipeline(device='cpu')
    print(f"Pipeline active. ONNX active={pipeline.use_onnx}, model path={pipeline.codeformer_onnx_path}")
    
    # 2. Test Presets (A/B Test Modes)
    presets = [
        'Pure Quality',
        'Modern Portrait',
        'Old Photo Restoration',
        'Game / Anime Character'
    ]
    
    for preset in presets:
        print(f"\n[A/B Test Preset] Testing with Preset Mode: '{preset}'...")
        res_img = pipeline.process_image(
            img,
            w=0.85 if preset == 'Pure Quality' else 0.6,
            upscale=2,
            face_upsample=False,
            blend_softness=0.5,
            detection_model='retinaface_mobile0.25',
            preset_mode=preset,
            enable_super_clarity=True,
            clarity_strength=0.45,
            sharpen_amount=0.2
        )
        assert res_img is not None, f"Failed to process image with preset {preset}"
        print(f"  -> Preset '{preset}' Passed! Output shape: {res_img.shape}")

    # 3. Test Model Switching (A/B Model Engines)
    print("\n[A/B Test Models] Testing Model Engine Switching...")
    avail_models = pipeline.get_available_models()
    print(f"  Discovered models: {list(avail_models.keys())}")
    
    for model_name, model_path in list(avail_models.items())[:2]:
        print(f"  Testing model engine: '{model_name}'...")
        res_m = pipeline.process_image(
            img,
            w=0.8,
            upscale=2,
            model_version=model_name,
            detection_model='retinaface_mobile0.25'
        )
        assert res_m is not None, f"Failed with model {model_name}"
        print(f"  -> Model '{model_name}' Passed! Output shape: {res_m.shape}")

    print("\n" + "=" * 60)
    print("  SUCCESS: Phase 7 A/B Model & Preset Pipeline test passed! (Exit Code 0)")
    print("=" * 60)

if __name__ == "__main__":
    test_ab_ui_pipeline()
