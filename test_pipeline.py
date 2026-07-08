import os
import cv2
import sys
from pipeline import LocalAIEnhancerPipeline

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(project_dir, "models", "CodeFormer", "inputs", "whole_imgs", "00.jpg")
    output_path = os.path.join(project_dir, "test_output.png")
    
    print(f"Loading input image: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: Test image not found at {input_path}")
        sys.exit(1)
        
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not read image.")
        sys.exit(1)
        
    print(f"Input image shape: {img.shape}")
    
    try:
        # Initialize pipeline
        pipeline = LocalAIEnhancerPipeline(device='cpu') # Use CPU to run stably on test env
        
        # Test enhancement with w = 0.5 and blend_softness = 0.5
        print("Processing image with w=0.5 and blend_softness=0.5...")
        upscale = 2
        enhanced_img = pipeline.process_image(
            img, 
            w=0.5, 
            detection_model='retinaface_resnet50', 
            upscale=upscale, 
            blend_softness=0.5
        )
        
        print(f"Enhanced image shape: {enhanced_img.shape}")
        
        # Save output image
        cv2.imwrite(output_path, enhanced_img)
        print(f"Saved enhanced image to: {output_path}")
        
        # Calculate expected output dimensions (using round to match OpenCV's resizing)
        min_dim = min(img.shape[:2])
        scale_factor = 512.0 / min_dim if min_dim < 512 else 1.0
        expected_h = int(round(img.shape[0] * scale_factor)) * upscale
        expected_w = int(round(img.shape[1] * scale_factor)) * upscale
        
        print(f"Expected output shape: ({expected_h}, {expected_w}, 3)")
        assert enhanced_img.shape[0] == expected_h, f"Output height mismatch: got {enhanced_img.shape[0]}, expected {expected_h}"
        assert enhanced_img.shape[1] == expected_w, f"Output width mismatch: got {enhanced_img.shape[1]}, expected {expected_w}"
        
        print("\nSUCCESS: Custom hybrid pipeline tested successfully.")
        
    except Exception as e:
        print(f"\nFAILURE: Error during pipeline test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
