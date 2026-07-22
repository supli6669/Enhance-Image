import os
import sys
import cv2

# Ensure project root is on sys.path
tools_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(tools_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from pipeline import LocalAIEnhancerPipeline

def test_ab_ui_pipeline():
    print("=== Phase 7: A/B Test & Multi-Model Pipeline Verification ===")
    
    input_path = os.path.join(project_dir, "models", "CodeFormer", "inputs", "whole_imgs", "00.jpg")
    print(f"Loading test image: {input_path}")
    img = cv2.imread(input_path)
    assert img is not None, "Failed to load test image!"
    
    print(f"Input image loaded successfully, shape: {img.shape}")
    
    # 1. Initialize Pipeline
    pipeline = LocalAIEnhancerPipeline(device='cpu')
    print(f"Pipeline active. ONNX active={pipeline.use_onnx}, model path={pipeline.codeformer_onnx_path}")
    
    # 2. Test Presets (A/B Test Modes)
    presets = ['default', 'portrait', 'old_photo', 'game_character']
    
    for preset in presets:
        print(f"\nTesting Pipeline Execution with Preset Mode: '{preset}'...")
        res_img, face_count = pipeline.process_image(
            img,
            w=0.6,
            upscale=2,
            face_upsample=False,
            blend_softness=0.5,
            detection_model='retinaface_mobile0.25',
            preset_mode=preset,
            enable_eyes=True,
            enable_lips=True,
            enable_skin=True,
            sharpen_amount=0.2
        )
        assert res_img is not None, f"Failed to process image with preset {preset}"
        print(f"  Preset '{preset}' Passed! Detected faces: {face_count}, Output shape: {res_img.shape}")
        
    print("\nSUCCESS: Phase 7 A/B Model & Preset Pipeline test completed with exit code 0!")

if __name__ == "__main__":
    test_ab_ui_pipeline()
