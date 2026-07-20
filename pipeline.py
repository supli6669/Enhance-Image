import os
import sys
import cv2
import numpy as np
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms.functional import normalize

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

def _get_ort_providers():
    if not HAS_ONNX:
        return []
    try:
        available = ort.get_available_providers()
        preferred = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = [p for p in preferred if p in available]
        return providers if providers else ['CPUExecutionProvider']
    except Exception:
        return ['CPUExecutionProvider']


# Ensure CodeFormer and tools directories are on sys.path
project_dir = os.path.dirname(os.path.abspath(__file__))
codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")
tools_dir = os.path.join(project_dir, "tools")
for p in (codeformer_dir, tools_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper

class LocalAIEnhancerPipeline:
    def __init__(self, device=None, progress_callback=None):
        """Initialize the CodeFormer model and helper pipeline.
        
        Args:
            device: torch device ('cuda' or 'cpu')
            progress_callback: Optional callback function(stage, progress, message) for progress reporting
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.progress_callback = progress_callback
        self.cancel_flag = False
            
        print(f"[Pipeline] Initializing pipeline on device: {self.device}")
        
        # Report initialization progress
        self._report_progress("initialization", 0.1, "Loading CodeFormer model...")
        
        # Check if ONNX models exist and should be used
        base_cf = os.path.join(project_dir, "weights", "CodeFormer", "codeformer")
        onnx_candidates = [
            base_cf + "_int8_v2.onnx",
            base_cf + "_int8.onnx",
            base_cf + ".onnx"
        ]
        
        self.use_onnx = False
        self.ort_session_cf = None
        self.codeformer_onnx_path = None

        if HAS_ONNX:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            for candidate in onnx_candidates:
                if os.path.exists(candidate):
                    try:
                        print(f"[Pipeline] Attempting to load ONNX model: {candidate}")
                        session = ort.InferenceSession(candidate, sess_options=opts, providers=_get_ort_providers())
                        self.ort_session_cf = session
                        self.codeformer_onnx_path = candidate
                        self.use_onnx = True
                        print(f"[Pipeline] Successfully loaded ONNX model: {candidate}")
                        break
                    except Exception as e:
                        print(f"[Pipeline] Warning: Failed to load ONNX model {candidate}: {e}")
                        # If a candidate fails, continue trying the next candidate

        base_re = os.path.join(project_dir, "weights", "realesrgan", "realesrgan")
        self.realesrgan_onnx_path = base_re + "_int8.onnx" if os.path.exists(base_re + "_int8.onnx") else base_re + ".onnx"
        self.use_re_onnx = HAS_ONNX and os.path.exists(self.realesrgan_onnx_path)
        
        # Cache for ONNX sessions
        self._onnx_session_cache = {}
        
        if self.use_onnx:
            self.net = None
        else:
            print("[Pipeline] ONNX disabled or unavailable. Falling back to PyTorch model.")
            # Load CodeFormer network architecture
            self.net = ARCH_REGISTRY.get('CodeFormer')(
                dim_embd=512, 
                codebook_size=1024, 
                n_head=8, 
                n_layers=9, 
                connect_list=['32', '64', '128', '256']
            ).to(self.device)
            
            # Load weights
            weights_path = os.path.join(project_dir, "weights", "CodeFormer", "codeformer.pth")
            if not os.path.exists(weights_path):
                print("[Pipeline] Pretrained weights not found. Automatically downloading models...")
                try:
                    import download_weights
                    download_weights.main()
                except Exception as e:
                    print(f"[Pipeline] Error during automatic weight download: {e}")
                    raise FileNotFoundError(f"CodeFormer weights not found at {weights_path} and auto-download failed. Please run download_weights.py manually.")
                
            print(f"[Pipeline] Loading weights from {weights_path}...")
            checkpoint = torch.load(weights_path, map_location=self.device)
            if 'params_ema' in checkpoint:
                self.net.load_state_dict(checkpoint['params_ema'])
            else:
                self.net.load_state_dict(checkpoint['params'])
            self.net.eval()
            print("[Pipeline] CodeFormer model loaded successfully.")
            
        # Cache for FaceRestoreHelper instances
        self._face_helper_cache = {}
        
        # Threading lock for concurrent ONNX inference sessions
        self.cf_onnx_lock = threading.Lock()

        # Report initialization complete
        self._report_progress("initialization", 1.0, "Pipeline ready!")
    
    def _report_progress(self, stage, progress, message):
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback(stage, progress, message)
    
    def _check_cancelled(self):
        """Check if processing was cancelled by user."""
        return self.cancel_flag
    
    def _get_onnx_session(self, path, providers=None):
        """Get or create cached ONNX session."""
        if path not in self._onnx_session_cache:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if providers is None:
                providers = _get_ort_providers()
            self._onnx_session_cache[path] = ort.InferenceSession(path, sess_options=opts, providers=providers)
        return self._onnx_session_cache[path]
    def _enhance_realesrgan_onnx_single(self, img, upscale):
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_rgb, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)

        # B8 FIX: Use unified _get_onnx_session() cache instead of ad-hoc
        # hasattr/None check, which would not survive garbage collection.
        session = self._get_onnx_session(self.realesrgan_onnx_path)

        ort_inputs = {session.get_inputs()[0].name: img_input}
        ort_outs = session.run(None, ort_inputs)
        output_tensor = ort_outs[0]
        
        output = np.squeeze(output_tensor, axis=0)
        output = np.clip(output, 0, 1)
        output = np.transpose(output, (1, 2, 0))
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output_bgr = (output_bgr * 255.0).round().astype(np.uint8)
        
        if upscale != 2:
            output_bgr = cv2.resize(output_bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_LANCZOS4)
            
        return output_bgr

    def enhance_realesrgan_onnx(self, img, upscale):
        h, w, c = img.shape
        tile_size = 400
        tile_pad = 40
        
        # If the image is small enough, run single inference directly
        if h <= tile_size and w <= tile_size:
            return self._enhance_realesrgan_onnx_single(img, upscale)
            
        print(f"[Pipeline] Image dimensions {w}x{h} exceed tile size {tile_size}. Running tile-based ONNX upscaling...")
        
        # We perform tiles at scale=2 since the model is 2x, then resize final stitched image if upscale != 2
        output_h, output_w = h * 2, w * 2
        output_img = np.zeros((output_h, output_w, c), dtype=np.uint8)
        
        stride = tile_size - tile_pad * 2
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Bounding box of the original crop (with overlap padding)
                y1 = max(0, y - tile_pad)
                x1 = max(0, x - tile_pad)
                y2 = min(h, y + tile_size - tile_pad)
                x2 = min(w, x + tile_size - tile_pad)
                
                tile = img[y1:y2, x1:x2]
                
                # Inference tile at 2x
                enhanced_tile = self._enhance_realesrgan_onnx_single(tile, 2)
                
                # Stitch back by calculating crop regions to drop the overlap padding
                pad_top = y - y1
                pad_left = x - x1
                
                w_crop = min(stride, w - x)
                h_crop = min(stride, h - y)
                
                # Target coordinates in output_img
                oy1, ox1 = y * 2, x * 2
                oy2, ox2 = (y + h_crop) * 2, (x + w_crop) * 2
                
                # Source coordinates in enhanced_tile (compensating for pad_top/pad_left)
                ty1, tx1 = pad_top * 2, pad_left * 2
                ty2, tx2 = (pad_top + h_crop) * 2, (pad_left + w_crop) * 2
                
                output_img[oy1:oy2, ox1:ox2] = enhanced_tile[ty1:ty2, tx1:tx2]
                
        if upscale != 2:
            output_img = cv2.resize(output_img, (w * upscale, h * upscale), interpolation=cv2.INTER_LANCZOS4)
            
        return output_img
    def run_onnx_batch(self, faces_np, w_val):
        """Helper to run ONNX batch inference."""
        w_np = np.full((faces_np.shape[0],), w_val, dtype=np.float32)
        ort_inputs = {
            self.ort_session_cf.get_inputs()[0].name: faces_np,
            self.ort_session_cf.get_inputs()[1].name: w_np
        }
        with self.cf_onnx_lock:
            ort_outs = self.ort_session_cf.run(None, ort_inputs)
        return ort_outs[0]
    def process_image(self, img, w=0.5, detection_model='retinaface_mobile0.25', upscale=2, blend_softness=0.5, bg_upsampler=None, det_threshold=0.5, sharpen_amount=0.0, face_upsample=False, batch_size=0, parallel=False, face_restore=True):
        """
        Enhance an image using the local CodeFormer pipeline.
        
        Args:
            img (numpy.ndarray): Input image in BGR format (OpenCV default).
            w (float): Fidelity weight (0.0 to 1.0). 0.0 for max quality, 1.0 for max fidelity.
            detection_model (str): Face detector model ('retinaface_mobile0.25', etc.).
            upscale (int): Upscale factor for output image.
            blend_softness (float): Blending mask softness (0.0 to 1.0).
            bg_upsampler (str): 'realesrgan' or None.
            det_threshold (float): Face detection confidence threshold.
            batch_size (int): Number of faces to process at once.
            face_restore (bool): Whether to perform face restoration.
            
        Returns:
            numpy.ndarray: Enhanced output image in BGR format.
        """
        # 1. Handle background upsampling first
        bg_img = None
        if bg_upsampler == 'realesrgan':
            self._report_progress("background", 0.1, "Upscaling background with Real-ESRGAN...")
            if self.use_re_onnx:
                print("[Pipeline] Running Real-ESRGAN background super-resolution using ONNX Runtime...")
                bg_img = self.enhance_realesrgan_onnx(img, upscale)
                self._report_progress("background", 0.5, "Background upscaled")
            else:
                if not hasattr(self, 'bg_upsampler_instance') or self.bg_upsampler_instance is None:
                    print("[Pipeline] Loading Real-ESRGAN background upsampler...")
                    realesrgan_path = os.path.join(project_dir, "weights", "realesrgan", "RealESRGAN_x2plus.pth")
                    if not os.path.exists(realesrgan_path):
                        print("[Pipeline] Real-ESRGAN weights not found. Automatically downloading...")
                        try:
                            import download_weights
                            download_weights.download_file("https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth", realesrgan_path)
                        except Exception as e:
                            print(f"[Pipeline] Error downloading Real-ESRGAN weights: {e}")
                            raise FileNotFoundError(f"Real-ESRGAN weights not found at {realesrgan_path} and auto-download failed.")
                    
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from basicsr.utils.realesrgan_utils import RealESRGANer
                    
                    use_half = False
                    if self.device.type == 'cuda':
                        no_half_gpu_list = ['1650', '1660']
                        if not any(gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list):
                            use_half = True
                            
                    model = RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32,
                        scale=2
                    )
                    self.bg_upsampler_instance = RealESRGANer(
                        scale=2,
                        model_path=realesrgan_path,
                        model=model,
                        tile=400,
                        tile_pad=40,
                        pre_pad=0,
                        half=use_half
                    )
                
                print("[Pipeline] Running Real-ESRGAN background super-resolution...")
                bg_img = self.bg_upsampler_instance.enhance(img, outscale=upscale)[0]
                self._report_progress("background", 0.5, "Background upscaled")

        if not face_restore:
            self._report_progress("complete", 1.0, "Enhancement complete!")
            if bg_img is not None:
                # Apply sharpening if requested
                if sharpen_amount > 0.0:
                    blurred = cv2.GaussianBlur(bg_img, (0, 0), 3.0)
                    bg_img = cv2.addWeighted(bg_img, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
                    bg_img = np.clip(bg_img, 0, 255).astype(np.uint8)
                return bg_img
            h, w_img, _ = img.shape
            resized = cv2.resize(img, (w_img * upscale, h * upscale), interpolation=cv2.INTER_LANCZOS4)
            if sharpen_amount > 0.0:
                blurred = cv2.GaussianBlur(resized, (0, 0), 3.0)
                resized = cv2.addWeighted(resized, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
                resized = np.clip(resized, 0, 255).astype(np.uint8)
            return resized

        # Set up FaceRestoreHelper for face processing
        os.environ['FACE_DETECTOR_PATH'] = os.path.join(project_dir, "weights", "facelib")
        # B3 FIX: upscale is NOT part of FaceRestoreHelper initialisation — it only
        # affects warpAffine geometry in paste_faces_custom_blend. Including upscale
        # in the cache key caused a full model re-init (3-5 s) on every upscale
        # factor change. Only the detection model matters for the helper instance.
        cache_key = detection_model
        if cache_key not in self._face_helper_cache:
            print(f"[Pipeline] Creating new FaceRestoreHelper for {detection_model} (upscale={upscale})...")
            face_helper = FaceRestoreHelper(
                upscale,
                face_size=512,
                crop_ratio=(1, 1),
                det_model=detection_model,
                save_ext='png',
                use_parse=True,
                device=self.device
            )
            
            # Modify confidence threshold dynamically on the underlying detector
            if hasattr(face_helper, 'face_detector'):
                detector = face_helper.face_detector
                if hasattr(detector, 'detect_faces'):
                    original_detect_faces = detector.detect_faces
                    def custom_detect_faces(image, *args, **kwargs):
                        detector_class = detector.__class__.__name__
                        thresh = getattr(detector, 'custom_det_threshold', 0.5)
                        if "Yolo" in detector_class:
                            kwargs['conf_thres'] = thresh
                        else:
                            kwargs['conf_threshold'] = thresh
                        return original_detect_faces(image, *args, **kwargs)
                    detector.detect_faces = custom_detect_faces
            self._face_helper_cache[cache_key] = face_helper
        else:
            face_helper = self._face_helper_cache[cache_key]
            
        # Update threshold dynamically
        if hasattr(face_helper, 'face_detector'):
            face_helper.face_detector.custom_det_threshold = det_threshold

        face_helper.clean_all()
        face_helper.read_image(img)
        
        # 2. Detect face landmarks and align/crop faces
        self._report_progress("detection", 0.1, f"Detecting faces with {detection_model}...")
        print(f"[Pipeline] Running face detection model: {detection_model} with threshold: {det_threshold}...")
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=False, 
            resize=640, 
            eye_dist_threshold=5
        )
        print(f"[Pipeline] Detected {num_det_faces} faces.")
        self._report_progress("detection", 0.3, f"Detected {num_det_faces} faces")

        if num_det_faces == 0:
            if bg_img is not None:
                return bg_img
            # Return resized background if no faces are detected and no AI upscaler used
            h, w_img, _ = img.shape
            return cv2.resize(img, (w_img * upscale, h * upscale), interpolation=cv2.INTER_LANCZOS4)
            
        face_helper.align_warp_face()
        
        # 2. Process each cropped face through CodeFormer
        self._report_progress("restoration", 0.1, f"Restoring {num_det_faces} face(s)...")
        if batch_size > 1 and self.use_onnx:
            faces_t = []
            for cropped_face in face_helper.cropped_faces:
                cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                faces_t.append(cropped_face_t)
            faces_np = torch.stack(faces_t).numpy()
            
            # Process in batches
            all_restored = []
            for i in range(0, len(faces_np), batch_size):
                batch = faces_np[i:i+batch_size]
                out_batch = self.run_onnx_batch(batch, w)
                all_restored.append(out_batch)
            
            output = np.concatenate(all_restored, axis=0)
            for i in range(output.shape[0]):
                res = np.squeeze(output[i], axis=0)
                res = np.clip(res, -1.0, 1.0)
                res = (res + 1.0) / 2.0 * 255.0
                res = np.transpose(res, (1, 2, 0))
                face_helper.add_restored_face(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_RGB2BGR), face_helper.cropped_faces[i])
            
            self._report_progress("restoration", 0.8, "Face restoration complete")
        else:
            # B2 FIX: Only use ThreadPoolExecutor when there are multiple faces.
            # For single-face images (the common case), spawning a thread pool
            # adds ~20 ms of overhead with zero parallelism benefit.
            if parallel and len(face_helper.cropped_faces) > 1:
                def _process_face(idx, cropped_face):
                    if self.use_onnx:
                        try:
                            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                            cropped_face_np = cropped_face_t.unsqueeze(0).numpy()
                            output = self.run_onnx_batch(cropped_face_np, w)
                            output = np.squeeze(output, axis=0)
                            output = np.clip(output, -1.0, 1.0)
                            output = (output + 1.0) / 2.0 * 255.0
                            output = np.transpose(output, (1, 2, 0))
                            restored = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
                        except Exception as error:
                            print(f"[Pipeline] Failed CodeFormer ONNX inference for face index {idx}: {error}")
                            restored = cropped_face.copy()
                    else:
                        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                        cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
                        try:
                            with torch.no_grad():
                                output = self.net(cropped_face_t, w=w, adain=True)[0]
                                restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                        except Exception as error:
                            print(f"[Pipeline] Failed CodeFormer inference for face index {idx}: {error}")
                            restored = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
                    restored = restored.astype('uint8')
                    return idx, restored

                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(lambda args: _process_face(*args), enumerate(face_helper.cropped_faces)))
                for idx, restored_face in sorted(results):
                    face_helper.add_restored_face(restored_face, face_helper.cropped_faces[idx])
            else:
                for idx, cropped_face in enumerate(face_helper.cropped_faces):
                    if self.use_onnx:
                        try:
                            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                            cropped_face_np = cropped_face_t.unsqueeze(0).numpy()
                            output = self.run_onnx_batch(cropped_face_np, w)
                            output = np.squeeze(output, axis=0)
                            output = np.clip(output, -1.0, 1.0)
                            output = (output + 1.0) / 2.0 * 255.0
                            output = np.transpose(output, (1, 2, 0))
                            restored = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
                        except Exception as error:
                            print(f"[Pipeline] Failed CodeFormer ONNX inference for face index {idx}: {error}")
                            restored = cropped_face.copy()
                    else:
                        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                        cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
                        try:
                            with torch.no_grad():
                                output = self.net(cropped_face_t, w=w, adain=True)[0]
                                restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                        except Exception as error:
                            print(f"[Pipeline] Failed CodeFormer inference for face index {idx}: {error}")
                            restored = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
                    restored = restored.astype('uint8')
                    face_helper.add_restored_face(restored, cropped_face)
            
            self._report_progress("restoration", 0.8, "Face restoration complete")
            
        # 3. Paste restored faces back into input image with custom soft blending
        self._report_progress("blending", 0.1, f"Blending {len(face_helper.restored_faces)} face(s)...")
        print(f"[Pipeline] Seamlessly pasting {len(face_helper.restored_faces)} restored faces back...")
        face_helper.get_inverse_affine(None)
        
        enhanced_img = self.paste_faces_custom_blend(
            face_helper, 
            upscale=upscale, 
            blend_softness=blend_softness,
            bg_img=bg_img,
            sharpen_amount=sharpen_amount,
            face_upsample=face_upsample,
            w=w
        )
        
        self._report_progress("blending", 1.0, "Blending complete!")
        self._report_progress("complete", 1.0, "Enhancement complete!")
        
        return enhanced_img

    def paste_faces_custom_blend(self, face_helper, upscale, blend_softness, bg_img=None, sharpen_amount=0.0, face_upsample=False, w=0.5):
        """Custom implementation of face pasting with adjustable soft blending mask."""
        h, w_img, _ = face_helper.input_img.shape
        h_up, w_up = int(h * upscale), int(w_img * upscale)
        
        # Normalize face size to tuple just in case it is an integer in some facexlib versions
        fs = face_helper.face_size
        raw_face_size = fs if isinstance(fs, tuple) else (fs, fs)
        
        # Initialize background image (upsampled background)
        if bg_img is None:
            upsample_img = cv2.resize(face_helper.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        else:
            upsample_img = cv2.resize(bg_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        
        for idx, (restored_face, inverse_affine) in enumerate(zip(face_helper.restored_faces, face_helper.inverse_affine_matrices)):
            inv_aff = inverse_affine.copy()
            cropped_face = face_helper.cropped_faces[idx]
            
            if upscale > 1:
                # Upscale the restored face using Real-ESRGAN to maintain super-resolution sharpness if enabled
                if face_upsample and self.use_re_onnx:
                    restored_face_up = self.enhance_realesrgan_onnx(restored_face, upscale)
                elif face_upsample and hasattr(self, 'bg_upsampler_instance') and self.bg_upsampler_instance is not None:
                    restored_face_up = self.bg_upsampler_instance.enhance(restored_face, outscale=upscale)[0]
                else:
                    # Fallback to Lanczos if no Real-ESRGAN instance loaded or face_upsample is disabled
                    restored_face_up = cv2.resize(restored_face, (raw_face_size[0] * upscale, raw_face_size[1] * upscale), interpolation=cv2.INTER_LANCZOS4)
                
                # Blend with original cropped face to preserve original high-resolution details when w > 0
                if w > 0.0:
                    original_face_up = cv2.resize(cropped_face, (raw_face_size[0] * upscale, raw_face_size[1] * upscale), interpolation=cv2.INTER_LANCZOS4)
                    restored_face_up = cv2.addWeighted(original_face_up, w, restored_face_up, 1.0 - w, 0.0)
                
                inv_aff /= upscale
                inv_aff[:, 2] *= upscale
                face_size = (raw_face_size[0] * upscale, raw_face_size[1] * upscale)
                inv_restored = cv2.warpAffine(restored_face_up, inv_aff, (w_up, h_up))
            else:
                # Blend with original cropped face to preserve original high-resolution details when w > 0
                if w > 0.0:
                    restored_face = cv2.addWeighted(cropped_face, w, restored_face, 1.0 - w, 0.0)
                
                # Add an offset to inverse affine matrix, for more precise back alignment
                extra_offset = 0
                inv_aff[:, 2] += extra_offset
                face_size = raw_face_size
                inv_restored = cv2.warpAffine(restored_face, inv_aff, (w_up, h_up))
            
            # Create boundary mask
            mask = np.ones(face_size, dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inv_aff, (w_up, h_up))
            
            # Erode slightly to remove absolute boundary black edges
            erosion_size = max(1, int(2 * upscale))
            inv_mask_erosion = cv2.erode(
                inv_mask, 
                np.ones((erosion_size, erosion_size), np.uint8)
            )
            
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            total_face_area = np.sum(inv_mask_erosion)
            
            # --- CUSTOM ADJUSTABLE SOFT MASK BLENDING ---
            # Default CodeFormer edge is total_face_area**0.5 / 20. We scale it with blend_softness.
            base_edge = int(total_face_area ** 0.5) // 20
            
            # Map blend_softness (0.0 - 1.0) to actual feather radius
            # 0.0 -> very small feathering (harder edge, raw paste)
            # 0.5 -> standard CodeFormer feathering
            # 1.0 -> double size feathering (extra soft blend)
            feather_radius = max(1, int(base_edge * 2 * blend_softness))
            
            # Additional erosion to pull the mask inside the face region
            inv_mask_center = cv2.erode(
                inv_mask_erosion, 
                np.ones((feather_radius, feather_radius), np.uint8)
            )
            
            # Blur the core mask to create the soft gradient
            blur_size = feather_radius * 2
            if blur_size % 2 == 0:
                blur_size += 1
                
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size, blur_size), 0)
            inv_soft_mask = inv_soft_mask[:, :, None]
            
            # Apply parsing mask (if segmenter is available and loaded)
            if face_helper.use_parse and hasattr(face_helper, 'face_parse'):
                face_input = cv2.resize(restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_input = img2tensor(face_input.astype('float32') / 255.0, bgr2rgb=True, float32=True)
                normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                face_input = torch.unsqueeze(face_input, 0).to(face_helper.device)
                
                with torch.no_grad():
                    out = face_helper.face_parse(face_input)[0]
                out = out.argmax(dim=1).squeeze().cpu().numpy()
                
                parse_mask = np.zeros(out.shape)
                MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                for p_idx, color in enumerate(MASK_COLORMAP):
                    parse_mask[out == p_idx] = color
                
                # Double Gaussian blur on parse mask
                parse_mask = cv2.GaussianBlur(parse_mask, (101, 101), 11)
                parse_mask = cv2.GaussianBlur(parse_mask, (101, 101), 11)
                
                # Remove black border artifacts
                thres = 10
                parse_mask[:thres, :] = 0
                parse_mask[-thres:, :] = 0
                parse_mask[:, :thres] = 0
                parse_mask[:, -thres:] = 0
                parse_mask = parse_mask / 255.0
                
                parse_mask = cv2.resize(parse_mask, face_size)
                parse_mask = cv2.warpAffine(parse_mask, inv_aff, (w_up, h_up), flags=3)
                inv_soft_parse_mask = parse_mask[:, :, None]
                # Squeeze to 2D and convert to standard contiguous float32 arrays to avoid stride/broadcast issues
                mask1 = np.ascontiguousarray(inv_soft_parse_mask.squeeze(), dtype=np.float32)
                mask2 = np.ascontiguousarray(inv_soft_mask.squeeze(), dtype=np.float32)
                
                # Intersect soft boundary mask with face feature parsing mask in 2D
                fuse_mask_2d = (mask1 < mask2).astype(np.float32)
                fuse_mask = fuse_mask_2d[:, :, None]
                
                # Ensure original masks are properly shaped in 3D
                if len(inv_soft_mask.shape) == 2:
                    inv_soft_mask = inv_soft_mask[:, :, None]
                if len(inv_soft_parse_mask.shape) == 2:
                    inv_soft_parse_mask = inv_soft_parse_mask[:, :, None]
                    
                inv_soft_mask = inv_soft_parse_mask * fuse_mask + inv_soft_mask * (1 - fuse_mask)
            
            # Merge restored face onto the background
            upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
            
        upsample_img = np.clip(upsample_img, 0, 255).astype(np.uint8)
        
        # Apply Post-Processing Sharpening Filter (Unsharp Masking) if requested
        if sharpen_amount > 0.0:
            # Gaussian blur for detail isolation
            blurred = cv2.GaussianBlur(upsample_img, (0, 0), 3.0)
            # Unsharp masking formula: sharpened = original + amount * (original - blurred)
            upsample_img = cv2.addWeighted(upsample_img, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
            upsample_img = np.clip(upsample_img, 0, 255).astype(np.uint8)
            
        return upsample_img
