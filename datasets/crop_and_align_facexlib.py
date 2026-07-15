import os
import sys
import cv2
import shutil
import torch

# Ensure project directories are in sys.path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")
if codeformer_dir not in sys.path:
    sys.path.insert(0, codeformer_dir)

from facelib.utils.face_restoration_helper import FaceRestoreHelper

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SRC_ROOT = os.path.join(project_dir, "datasets", "game_characters")
DST_ROOT = os.path.join(project_dir, "datasets", "ffhq", "ffhq_512")
DETECTOR = "retinaface_mobile0.25"  # Fast CPU detector
BLUR_THRESHOLD = 80.0  # Discard crops with Laplacian variance below this

def is_blurry(img, threshold=BLUR_THRESHOLD):
    """Check if an image is blurry using the variance of Laplacian method."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Align] Using device: {device}")
    
    # Configure face detector path
    os.environ['FACE_DETECTOR_PATH'] = os.path.join(project_dir, "weights", "facelib")
    
    # Clear and recreate destination folder
    if os.path.exists(DST_ROOT):
        shutil.rmtree(DST_ROOT)
    os.makedirs(DST_ROOT, exist_ok=True)
    
    # Initialize face helper
    face_helper = FaceRestoreHelper(
        1,  # upscale factor
        face_size=512,
        crop_ratio=(1, 1),
        det_model=DETECTOR,
        save_ext='png',
        use_parse=False,
        device=device
    )
    
    # Gather all raw images
    raw_images = []
    for root, _, files in os.walk(SRC_ROOT):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                raw_images.append(os.path.join(root, f))
                
    print(f"[Align] Found {len(raw_images)} raw images to process.")
    
    count = 0
    failures = 0
    skipped_blur = 0
    
    for i, img_path in enumerate(raw_images):
        try:
            img = cv2.imread(img_path)
            if img is None:
                failures += 1
                continue
                
            face_helper.clean_all()
            face_helper.read_image(img)
            
            # Detect landmarks
            num_faces = face_helper.get_face_landmarks_5(
                only_center_face=False,
                resize=640,
                eye_dist_threshold=5
            )
            
            if num_faces == 0:
                # No face detected, skip
                continue
                
            # Warp and align
            face_helper.align_warp_face()
            
            # Save cropped faces
            for cropped_face in face_helper.cropped_faces:
                # Blur filter check
                if is_blurry(cropped_face):
                    skipped_blur += 1
                    continue
                    
                count += 1
                dest_path = os.path.join(DST_ROOT, f"face_{count:06d}.png")
                cv2.imwrite(dest_path, cropped_face)
                
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(raw_images)} images. Extracted {count} faces (skipped {skipped_blur} blurry)...")
                
        except Exception as e:
            # print(f"Error processing {img_path}: {e}")
            failures += 1
            
    print(f"\n[DONE] Successfully processed all images!")
    print(f"  Total raw images: {len(raw_images)}")
    print(f"  Extracted face crops: {count}")
    print(f"  Skipped blurry crops: {skipped_blur}")
    print(f"  Failed loads: {failures}")
    print(f"  Destination: {DST_ROOT}\n")

if __name__ == "__main__":
    main()
