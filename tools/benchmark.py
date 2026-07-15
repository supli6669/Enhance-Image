import os
import cv2
import sys
import time

# Add project root to sys.path
tools_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(tools_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from pipeline import LocalAIEnhancerPipeline

def benchmark():
    input_path = os.path.join(project_dir, "models", "CodeFormer", "inputs", "whole_imgs", "00.jpg")
    print(f"Loading input image: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not read image.")
        return
        
    print(f"Input image shape: {img.shape}")
    
    print("\n--- Benchmark 1: Normal settings (CPU) ---")
    pipeline = LocalAIEnhancerPipeline(device='cpu')
    
    # Let's run a single process and print timing of subparts
    # To do this, let's copy process_image logic with timing
    
    # 1. Face detection timing
    t0 = time.time()
    os.environ['FACE_DETECTOR_PATH'] = os.path.join(project_dir, "weights", "facelib")
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    face_helper = FaceRestoreHelper(
        upscale_factor=2,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_mobile0.25',
        save_ext='png',
        use_parse=True,
        device=pipeline.device
    )
    face_helper.clean_all()
    face_helper.read_image(img)
    t1 = time.time()
    print(f"Helper init time: {t1 - t0:.4f}s")
    
    t0 = time.time()
    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=False, 
        resize=640, 
        eye_dist_threshold=5
    )
    t1 = time.time()
    print(f"Face detection (retinaface_resnet50) time: {t1 - t0:.4f}s (found {num_det_faces} faces)")
    
    # Detect with a faster model
    for faster_detector in ['retinaface_mobile0.25', 'YOLOv5n']:
        t0 = time.time()
        fh_fast = FaceRestoreHelper(
            upscale_factor=2,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=faster_detector,
            save_ext='png',
            use_parse=True,
            device=pipeline.device
        )
        fh_fast.clean_all()
        fh_fast.read_image(img)
        num_fast = fh_fast.get_face_landmarks_5(
            only_center_face=False, 
            resize=640, 
            eye_dist_threshold=5
        )
        t1 = time.time()
        print(f"Face detection ({faster_detector}) time: {t1 - t0:.4f}s (found {num_fast} faces)")
        
    face_helper.align_warp_face()
    
    # 2. CodeFormer inference timing
    t0 = time.time()
    cropped_face = face_helper.cropped_faces[0]
    import numpy as np
    from basicsr.utils import img2tensor
    from torchvision.transforms.functional import normalize
    import onnxruntime as ort
    
    cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_np = cropped_face_t.unsqueeze(0).numpy()
    
    w_np = np.array([0.5], dtype=np.float32)
    ort_inputs = {
        pipeline.ort_session_cf.get_inputs()[0].name: cropped_face_np,
        pipeline.ort_session_cf.get_inputs()[1].name: w_np
    }
    ort_outs = pipeline.ort_session_cf.run(None, ort_inputs)
    t1 = time.time()
    print(f"CodeFormer ONNX inference time (1 face): {t1 - t0:.4f}s")
    
    # 3. Real-ESRGAN Background Upscaling timing (optional, disabled by default)
    run_re_bg = False
    if run_re_bg:
        t0 = time.time()
        bg_img = pipeline.enhance_realesrgan_onnx(img, 2)
        t1 = time.time()
        print(f"Real-ESRGAN Background Upscaling time: {t1 - t0:.4f}s")
    
    # 4. Real-ESRGAN Face Upscaling timing (optional, disabled by default)
    run_re_face = False
    if run_re_face:
        t0 = time.time()
        face_up_re = pipeline.enhance_realesrgan_onnx(cropped_face, 2)
        t1 = time.time()
        print(f"Real-ESRGAN Face Upscaling time (512x512 -> 1024x1024): {t1 - t0:.4f}s")
    
    # 5. Lanczos Face Upscaling timing
    t0 = time.time()
    face_up_lanczos = cv2.resize(cropped_face, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
    t1 = time.time()
    print(f"Lanczos Face Upscaling time (512x512 -> 1024x1024): {t1 - t0:.4f}s")

if __name__ == "__main__":
    benchmark()
