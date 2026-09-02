import os
import sys
import gc
import time
import cv2
import numpy as np
import torch
import threading
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms.functional import normalize

try:
    import tools.compat_shim
except Exception:
    pass

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

def _get_ort_providers():
    """Return a list of working ONNX Runtime providers, probing each one first
    to avoid DLL-not-found errors (e.g. OpenVINO) being printed to stderr."""
    if not HAS_ONNX:
        return ['CPUExecutionProvider']
    try:
        available = set(ort.get_available_providers())
        preferred = ['DmlExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        # Only include providers that are actually available and skip OpenVINO
        # (it registers as available but the openvino.dll is missing on most
        # local installs, causing loud stderr errors that confuse test runners)
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
from wink_enhancer import WinkQualityEnhancer

def get_available_models(weights_dir=None):
    """Return dictionary of available CodeFormer models found in weights directory."""
    if weights_dir is None:
        weights_dir = os.path.join(project_dir, "weights", "CodeFormer")
    models = {}
    if not os.path.exists(weights_dir):
        return models
    
    candidates = [
        ("codeformer_int8_v3.onnx", "CodeFormer v3.0 (Fast INT8 ArcFace) [⚡ Recommended]"),
        ("codeformer_v3.onnx", "CodeFormer v3.0 (FP32 ArcFace High-Identity)"),
        ("codeformer_int8_v2.onnx", "CodeFormer v2.0 (Fast INT8 Quantized)"),
        ("codeformer_int8.onnx", "CodeFormer v1.0 (Fast INT8 Baseline)"),
        ("codeformer.onnx", "CodeFormer Standard ONNX (FP32)"),
        ("codeformer.pth", "CodeFormer PyTorch Native (FP32 Weights)"),
    ]
    for filename, label in candidates:
        full_path = os.path.join(weights_dir, filename)
        if os.path.isfile(full_path):
            models[label] = full_path
    return models

# The Streamlit app caches one pipeline instance. Keep callback state in the
# calling context rather than on that shared instance so progress cannot leak
# between users/requests.
_active_progress_callback = ContextVar("active_progress_callback", default=None)

class LocalAIEnhancerPipeline:
    def __init__(self, device=None, progress_callback=None, model_path_override=None):
        """Initialize the CodeFormer model and helper pipeline.
        
        Args:
            device: torch device ('cuda' or 'cpu')
            progress_callback: Optional callback function(stage, progress, message) for progress reporting
            model_path_override: Optional path to specific ONNX or PyTorch model
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.progress_callback = progress_callback
        self.cancel_flag = False
        self.project_dir = project_dir
        self.wink_enhancer = WinkQualityEnhancer()
            
        print(f"[Pipeline] Initializing pipeline on device: {self.device}")
        self._report_progress("initialization", 0.1, "Loading CodeFormer model...")
        
        base_cf = os.path.join(project_dir, "weights", "CodeFormer", "codeformer")
        onnx_candidates = [
            base_cf + "_int8_v3.onnx",
            base_cf + "_v3.onnx",
            base_cf + "_int8_v2.onnx",
            base_cf + "_int8.onnx",
            base_cf + ".onnx"
        ]
        if model_path_override and os.path.exists(model_path_override):
            onnx_candidates.insert(0, model_path_override)
        
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

        re_dir = os.path.join(project_dir, "weights", "realesrgan")
        re_candidates = [
            os.path.join(re_dir, "realesrgan_custom.onnx"),
            os.path.join(re_dir, "realesrgan_int8.onnx"),
            os.path.join(re_dir, "realesrgan.onnx"),
            os.path.join(project_dir, "realesrgan_custom.onnx")
        ]
        self.realesrgan_onnx_path = None
        for cand in re_candidates:
            if os.path.exists(cand):
                self.realesrgan_onnx_path = cand
                print(f"[Pipeline] Loaded Real-ESRGAN upscaler: {cand}")
                break
        self.use_re_onnx = HAS_ONNX and (self.realesrgan_onnx_path is not None)
        
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
        
        # Serialise a complete request: FaceRestoreHelper, the model caches and
        # post-processors all hold mutable per-image state.
        self._processing_lock = threading.RLock()

        # Threading lock for concurrent ONNX inference sessions
        self.cf_onnx_lock = threading.Lock()

        # Report initialization complete
        self._report_progress("initialization", 1.0, "Pipeline ready!")
    
    def _report_progress(self, stage, progress, message):
        """Report progress to callback if available."""
        callback = _active_progress_callback.get()
        if callback is None:
            callback = self._default_progress_callback
        if callback:
            callback(stage, progress, message)

    @property
    def progress_callback(self):
        """Legacy default callback; prefer ``process_image(..., progress_callback=...)``."""
        return self._default_progress_callback

    @progress_callback.setter
    def progress_callback(self, callback):
        self._default_progress_callback = callback
    
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
    def _enhance_realesrgan_onnx_single(self, img, upscale, model_path=None):
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_rgb, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)

        # Use passed model_path or default cached path
        active_path = model_path if (model_path and os.path.exists(model_path)) else self.realesrgan_onnx_path
        session = self._get_onnx_session(active_path)

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

    def enhance_realesrgan_onnx(self, img, upscale, model_path=None):
        h, w, c = img.shape
        tile_size = 400
        tile_pad = 40
        
        # If the image is small enough, run single inference directly
        if h <= tile_size and w <= tile_size:
            return self._enhance_realesrgan_onnx_single(img, upscale, model_path=model_path)
            
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
                enhanced_tile = self._enhance_realesrgan_onnx_single(tile, 2, model_path=model_path)
                
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
    def get_available_models(self):
        """Scan and return list of available CodeFormer models with user-friendly labels."""
        weights_dir = os.path.join(self.project_dir, "weights", "CodeFormer")
        models = {}
        
        # Candidate map
        v3_int8 = os.path.join(weights_dir, "codeformer_int8_v3.onnx")
        v3_fp32 = os.path.join(weights_dir, "codeformer_v3.onnx")
        v2_int8 = os.path.join(weights_dir, "codeformer_int8_v2.onnx")
        v1_fp32 = os.path.join(weights_dir, "codeformer.onnx")
        v1_pth = os.path.join(weights_dir, "codeformer.pth")
        
        if os.path.exists(v3_int8):
            models["🔥 CodeFormer v3.0 INT8 (ArcFace Cloud Trained)"] = v3_int8
        if os.path.exists(v3_fp32):
            models["🔥 CodeFormer v3.0 FP32 (ArcFace Cloud Trained)"] = v3_fp32
        if os.path.exists(v2_int8):
            models["⚡ CodeFormer v2.0 INT8 (Fast CPU Quantized)"] = v2_int8
        if os.path.exists(v1_fp32):
            models["💎 CodeFormer v1.0 FP32 (ONNX Baseline)"] = v1_fp32
        if os.path.exists(v1_pth):
            models["📦 CodeFormer Baseline (PyTorch .pth)"] = v1_pth
            
        return models

    def get_available_upscalers(self):
        """Scan and return dictionary of available Real-ESRGAN upscaler models."""
        upscalers = {}
        re_dir = os.path.join(self.project_dir, "weights", "realesrgan")
        custom_onnx = os.path.join(re_dir, "realesrgan_custom.onnx")
        int8_onnx = os.path.join(re_dir, "realesrgan_int8.onnx")
        base_onnx = os.path.join(re_dir, "realesrgan.onnx")

        if os.path.exists(custom_onnx):
            upscalers["🚀 Real-ESRGAN Custom ONNX (Universal Mới Train)"] = custom_onnx
        if os.path.exists(int8_onnx):
            upscalers["⚡ Real-ESRGAN INT8 ONNX (Siêu Tốc CPU)"] = int8_onnx
        if os.path.exists(base_onnx):
            upscalers["💎 Real-ESRGAN x2plus ONNX (Chuẩn Xintao)"] = base_onnx
            
        upscalers["⚡ Lanczos Fast CPU (Không Dùng AI)"] = "lanczos"
        return upscalers

    def run_onnx_batch(self, faces_np, w_val, session_override=None):
        """Helper to run ONNX batch inference."""
        session = session_override if session_override is not None else self.ort_session_cf
        w_np = np.full((faces_np.shape[0],), w_val, dtype=np.float32)
        ort_inputs = {
            session.get_inputs()[0].name: faces_np,
            session.get_inputs()[1].name: w_np
        }
        with self.cf_onnx_lock:
            ort_outs = session.run(None, ort_inputs)
        return ort_outs[0]

    def process_image(self, img, w=0.5, detection_model='retinaface_mobile0.25', upscale=2, blend_softness=0.5, bg_upsampler=None, det_threshold=0.5, sharpen_amount=0.0, face_upsample=False, batch_size=0, parallel=False, face_restore=True, wink_mode=True, eye_enhancement=True, skin_grain=0.15, color_match=True, enable_eyes=True, enable_lips=True, enable_skin=True, enable_teeth=True, enable_tone_glow=True, enable_dark_circles=True, enable_catchlight=True, catchlight_strength=0.55, enable_hair=True, hair_clarity=0.35, hair_sheen=0.25, enable_relighting=True, relighting_rim=0.25, relighting_tzone=0.20, enable_anti_glare=True, anti_glare_strength=0.50, enable_makeup=True, blush_strength=0.30, eyebrow_boost=0.35, enable_crystal_skin=True, crystal_skin_strength=0.45, enable_glossy_lips=True, lip_gloss=0.40, lip_vibrance=0.25, enable_doll_eye=True, doll_eye_depth=0.45, enable_golden_hour=False, golden_warmth=0.25, golden_bloom=0.20, enable_super_clarity=True, clarity_strength=0.40, enable_deblur=False, deblur_strength=0.35, enable_dehaze=True, dehaze_strength=0.25, color_lut="None", lut_intensity=1.0, bokeh_strength=0.0, preset_mode='Custom', chromatic_aberration=False, model_version='Auto', bg_upsampler_model=None, progress_callback=None):

        """Enhance one image without sharing request-specific state.

        ``progress_callback`` is scoped to this call.  The constructor callback
        remains supported as a legacy default for code that already uses it.
        """
        callback = self._default_progress_callback if progress_callback is None else progress_callback
        with self._processing_lock:
            callback_token = _active_progress_callback.set(callback)
            try:
                return self._process_image(
                    img, w, detection_model, upscale, blend_softness, bg_upsampler,
                    det_threshold, sharpen_amount, face_upsample, batch_size, parallel,
                    face_restore, wink_mode, eye_enhancement, skin_grain, color_match,
                    enable_eyes, enable_lips, enable_skin, enable_teeth, enable_tone_glow,
                    enable_dark_circles, enable_catchlight, catchlight_strength,
                    enable_hair, hair_clarity, hair_sheen,
                    enable_relighting, relighting_rim, relighting_tzone,
                    enable_anti_glare, anti_glare_strength,
                    enable_makeup, blush_strength, eyebrow_boost,
                    enable_crystal_skin, crystal_skin_strength,
                    enable_glossy_lips, lip_gloss, lip_vibrance,
                    enable_doll_eye, doll_eye_depth,
                    enable_golden_hour, golden_warmth, golden_bloom,
                    enable_super_clarity, clarity_strength,
                    enable_deblur, deblur_strength,
                    enable_dehaze, dehaze_strength,
                    color_lut, lut_intensity, bokeh_strength,
                    preset_mode, chromatic_aberration,
                    model_version, bg_upsampler_model
                )
            finally:
                _active_progress_callback.reset(callback_token)

    def _process_image(self, img, w=0.5, detection_model='retinaface_mobile0.25', upscale=2, blend_softness=0.5, bg_upsampler=None, det_threshold=0.5, sharpen_amount=0.0, face_upsample=False, batch_size=0, parallel=False, face_restore=True, wink_mode=True, eye_enhancement=True, skin_grain=0.15, color_match=True, enable_eyes=True, enable_lips=True, enable_skin=True, enable_teeth=True, enable_tone_glow=True, enable_dark_circles=True, enable_catchlight=True, catchlight_strength=0.55, enable_hair=True, hair_clarity=0.35, hair_sheen=0.25, enable_relighting=True, relighting_rim=0.25, relighting_tzone=0.20, enable_anti_glare=True, anti_glare_strength=0.50, enable_makeup=True, blush_strength=0.30, eyebrow_boost=0.35, enable_crystal_skin=True, crystal_skin_strength=0.45, enable_glossy_lips=True, lip_gloss=0.40, lip_vibrance=0.25, enable_doll_eye=True, doll_eye_depth=0.45, enable_golden_hour=False, golden_warmth=0.25, golden_bloom=0.20, enable_super_clarity=True, clarity_strength=0.40, enable_deblur=False, deblur_strength=0.35, enable_dehaze=True, dehaze_strength=0.25, color_lut="None", lut_intensity=1.0, bokeh_strength=0.0, preset_mode='Custom', chromatic_aberration=False, model_version='Auto', bg_upsampler_model=None):

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
            bg_upsampler_model (str): Optional path to specific Real-ESRGAN ONNX model.
            
        Returns:
            numpy.ndarray: Enhanced output image in BGR format.
        """
        # Apply Chromatic Aberration Correction if requested
        if chromatic_aberration and hasattr(self, 'wink_enhancer'):
            img = self.wink_enhancer.correct_chromatic_aberration(img)

        # Apply Preset parameters if specific preset mode is selected
        if preset_mode == 'Modern Portrait':
            w = 0.6
            wink_mode = True
            eye_enhancement = True
            skin_grain = 0.15
            color_match = True
            enable_eyes = True
            enable_lips = True
            enable_skin = True
        elif preset_mode == 'Old Photo Restoration':
            w = 0.85
            wink_mode = True
            eye_enhancement = True
            skin_grain = 0.05
            color_match = True
            enable_eyes = True
            enable_lips = True
            enable_skin = True
        elif preset_mode == 'Game / Anime Character':
            w = 0.3
            wink_mode = True
            eye_enhancement = False
            skin_grain = 0.0
            color_match = False
            enable_eyes = False
            enable_lips = False
            enable_skin = False

        # 1. Handle background upsampling first
        bg_img = None
        if bg_upsampler == 'realesrgan':
            self._report_progress("background", 0.1, "Upscaling image with Real-ESRGAN...")
            if self.use_re_onnx:
                print("[Pipeline] Running Real-ESRGAN super-resolution using ONNX Runtime...")
                bg_img = self.enhance_realesrgan_onnx(img, upscale, model_path=bg_upsampler_model)
                self._report_progress("background", 0.5, "Real-ESRGAN super-resolution complete")
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
        
        # Reset per-image helper state
        face_helper.clean_all()
        face_helper.read_image(img)
        
        # 2. Detect and align faces
        self._report_progress("detection", 0.1, f"Detecting faces with {detection_model}...")
        num_faces = face_helper.get_face_landmarks_5(
            only_center_face=False, 
            resize=640, 
            eye_dist_threshold=5
        )

        print(f"[Pipeline] Detected {num_faces} face(s).")
        self._report_progress("detection", 0.5, f"Detected {num_faces} face(s)")
        
        if num_faces == 0:
            print("[Pipeline] No faces detected in input image. Processing as Universal Image...")
            self._report_progress("enhancement", 0.5, "Enhancing full image (Real-ESRGAN / Clarity)...")
            if bg_img is not None:
                enhanced_img = bg_img
            elif self.use_re_onnx:
                enhanced_img = self.enhance_realesrgan_onnx(img, upscale)
            else:
                h, w_img, _ = img.shape
                enhanced_img = cv2.resize(img, (w_img * upscale, h * upscale), interpolation=cv2.INTER_LANCZOS4)

            # Apply Universal Clarity, Deblur, Dehaze, and Texture sharpening for non-face images
            if hasattr(self, 'wink_enhancer'):
                if enable_dehaze and dehaze_strength > 0.0:
                    enhanced_img = self.wink_enhancer.apply_dehaze_and_dynamic_contrast(enhanced_img, strength=dehaze_strength * 0.6)
                if enable_super_clarity and clarity_strength > 0.0:
                    enhanced_img = self.wink_enhancer.apply_laplacian_pyramid_clarity(enhanced_img, strength=clarity_strength * 0.45)
                if sharpen_amount > 0.0:
                    enhanced_img = self.wink_enhancer.unsharp_mask(enhanced_img, amount=sharpen_amount)

            self._report_progress("complete", 1.0, "Universal enhancement complete!")
            return enhanced_img
            
        face_helper.align_warp_face()
        if len(face_helper.cropped_faces) == 0:
            print("[Pipeline] No cropped faces extracted from detected landmarks. Processing as Universal Image...")
            self._report_progress("enhancement", 0.5, "Enhancing full image (Real-ESRGAN / Clarity)...")
            if bg_img is not None:
                enhanced_img = bg_img
            elif self.use_re_onnx:
                enhanced_img = self.enhance_realesrgan_onnx(img, upscale)
            else:
                h, w_img, _ = img.shape
                enhanced_img = cv2.resize(img, (w_img * upscale, h * upscale), interpolation=cv2.INTER_LANCZOS4)

            if hasattr(self, 'wink_enhancer'):
                if enable_dehaze and dehaze_strength > 0.0:
                    enhanced_img = self.wink_enhancer.apply_dehaze_and_dynamic_contrast(enhanced_img, strength=dehaze_strength * 0.6)
                if enable_super_clarity and clarity_strength > 0.0:
                    enhanced_img = self.wink_enhancer.apply_laplacian_pyramid_clarity(enhanced_img, strength=clarity_strength * 0.45)
                if sharpen_amount > 0.0:
                    enhanced_img = self.wink_enhancer.unsharp_mask(enhanced_img, amount=sharpen_amount)

            self._report_progress("complete", 1.0, "Universal enhancement complete!")
            return enhanced_img

        print(f"[Pipeline] Cropped {len(face_helper.cropped_faces)} face(s).")
        
        # Restore faces using CodeFormer model
        self._report_progress("restoration", 0.1, f"Restoring {len(face_helper.cropped_faces)} face(s) (w={w})...")
        
        # Resolve active model session override if specified
        active_session = None
        if model_version != 'Auto':
            avail_models = self.get_available_models()
            if model_version in avail_models:
                m_path = avail_models[model_version]
                if m_path.endswith('.onnx'):
                    active_session = self._get_onnx_session(m_path)

        # Process faces
        if parallel and len(face_helper.cropped_faces) > 1:
            print(f"[Pipeline] Processing {len(face_helper.cropped_faces)} faces in parallel...")
            def _process_face(idx, cropped_face):
                if self.use_onnx or active_session is not None:
                    try:
                        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                        cropped_face_np = cropped_face_t.unsqueeze(0).numpy()
                        output = self.run_onnx_batch(cropped_face_np, w, session_override=active_session)
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
                if self.use_onnx or active_session is not None:
                    try:
                        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                        cropped_face_np = cropped_face_t.unsqueeze(0).numpy()
                        output = self.run_onnx_batch(cropped_face_np, w, session_override=active_session)
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
            w=w,
            wink_mode=wink_mode,
            eye_enhancement=eye_enhancement,
            skin_grain=skin_grain,
            color_match=color_match,
            enable_eyes=enable_eyes,
            enable_lips=enable_lips,
            enable_skin=enable_skin,
            enable_teeth=enable_teeth,
            enable_tone_glow=enable_tone_glow,
            enable_dark_circles=enable_dark_circles,
            enable_catchlight=enable_catchlight,
            catchlight_strength=catchlight_strength,
            enable_hair=enable_hair,
            hair_clarity=hair_clarity,
            hair_sheen=hair_sheen,
            enable_relighting=enable_relighting,
            relighting_rim=relighting_rim,
            relighting_tzone=relighting_tzone,
            enable_anti_glare=enable_anti_glare,
            anti_glare_strength=anti_glare_strength,
            enable_makeup=enable_makeup,
            blush_strength=blush_strength,
            eyebrow_boost=eyebrow_boost,
            enable_crystal_skin=enable_crystal_skin,
            crystal_skin_strength=crystal_skin_strength,
            enable_glossy_lips=enable_glossy_lips,
            lip_gloss=lip_gloss,
            lip_vibrance=lip_vibrance,
            enable_doll_eye=enable_doll_eye,
            doll_eye_depth=doll_eye_depth,
            enable_golden_hour=enable_golden_hour,
            golden_warmth=golden_warmth,
            golden_bloom=golden_bloom,
            enable_super_clarity=enable_super_clarity,
            clarity_strength=clarity_strength,
            enable_deblur=enable_deblur,
            deblur_strength=deblur_strength,
            enable_dehaze=enable_dehaze,
            dehaze_strength=dehaze_strength
        )

        # 4. Apply Full-Image Multi-Scale Clarity & Crystal De-Haze if enabled
        if hasattr(self, 'wink_enhancer'):
            if enable_dehaze and dehaze_strength > 0.0:
                enhanced_img = self.wink_enhancer.apply_dehaze_and_dynamic_contrast(enhanced_img, strength=dehaze_strength * 0.6)
            if enable_super_clarity and clarity_strength > 0.0:
                enhanced_img = self.wink_enhancer.apply_laplacian_pyramid_clarity(enhanced_img, strength=clarity_strength * 0.45)
            if enable_golden_hour and (golden_warmth > 0.0 or golden_bloom > 0.0):
                enhanced_img = self.wink_enhancer.apply_golden_hour_glow(enhanced_img, warm_strength=golden_warmth * 0.6, bloom_strength=golden_bloom * 0.5)

        # 5. Apply Studio Optical Bokeh Blur if enabled
        if bokeh_strength > 0.0 and hasattr(self, 'wink_enhancer'):
            self._report_progress("postprocess", 0.8, f"Applying optical portrait bokeh (f/1.4 blur)...")
            enhanced_img = self.wink_enhancer.apply_portrait_bokeh(enhanced_img, face_bboxes=None, bokeh_strength=bokeh_strength)

        # 6. Apply Studio Cinematic Color LUT Grade if enabled
        if color_lut not in (None, "None", "Off", "") and hasattr(self, 'wink_enhancer'):
            self._report_progress("postprocess", 0.9, f"Applying cinematic {color_lut} LUT...")
            enhanced_img = self.wink_enhancer.apply_cinematic_lut(enhanced_img, lut_name=color_lut, intensity=lut_intensity)
        
        self._report_progress("blending", 1.0, "Blending complete!")
        self._report_progress("complete", 1.0, "Enhancement complete!")
        
        return enhanced_img

    def paste_faces_custom_blend(self, face_helper, upscale, blend_softness, bg_img=None, sharpen_amount=0.0, face_upsample=False, w=0.5, wink_mode=True, eye_enhancement=True, skin_grain=0.15, color_match=True, enable_eyes=True, enable_lips=True, enable_skin=True, enable_teeth=True, enable_tone_glow=True, enable_dark_circles=True, enable_catchlight=True, catchlight_strength=0.55, enable_hair=True, hair_clarity=0.35, hair_sheen=0.25, enable_relighting=True, relighting_rim=0.25, relighting_tzone=0.20, enable_anti_glare=True, anti_glare_strength=0.50, enable_makeup=True, blush_strength=0.30, eyebrow_boost=0.35, enable_crystal_skin=True, crystal_skin_strength=0.45, enable_glossy_lips=True, lip_gloss=0.40, lip_vibrance=0.25, enable_doll_eye=True, doll_eye_depth=0.45, enable_golden_hour=False, golden_warmth=0.25, golden_bloom=0.20, enable_super_clarity=True, clarity_strength=0.40, enable_deblur=False, deblur_strength=0.35, enable_dehaze=True, dehaze_strength=0.25):
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
            
            # Apply Wink-level quality post-processing (skin grain, eye sparkle, LAB tone balance)
            if wink_mode and hasattr(self, 'wink_enhancer'):
                parse_mask = None
                if hasattr(face_helper, 'face_parse') and face_helper.face_parse is not None:
                    try:
                        with torch.no_grad():
                            face_t = img2tensor(restored_face / 255.0, bgr2rgb=True, float32=True).unsqueeze(0).to(self.device)
                            normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                            out_parse = face_helper.face_parse(face_t)[0]
                            parse_mask = out_parse.argmax(dim=0).cpu().numpy()
                    except Exception:
                        parse_mask = None

                restored_face = self.wink_enhancer.enhance_face(
                    restored_face,
                    cropped_original=cropped_face,
                    parse_mask=parse_mask,
                    wink_mode=wink_mode,
                    eye_enhancement=eye_enhancement,
                    skin_grain=skin_grain,
                    color_match=color_match,
                    enable_eyes=enable_eyes,
                    enable_lips=enable_lips,
                    enable_skin=enable_skin,
                    enable_teeth=enable_teeth,
                    enable_tone_glow=enable_tone_glow,
                    enable_dark_circles=enable_dark_circles,
                    enable_catchlight=enable_catchlight,
                    catchlight_strength=catchlight_strength,
                    enable_hair=enable_hair,
                    hair_clarity=hair_clarity,
                    hair_sheen=hair_sheen,
                    enable_relighting=enable_relighting,
                    relighting_rim=relighting_rim,
                    relighting_tzone=relighting_tzone,
                    enable_anti_glare=enable_anti_glare,
                    anti_glare_strength=anti_glare_strength,
                    enable_makeup=enable_makeup,
                    blush_strength=blush_strength,
                    eyebrow_boost=eyebrow_boost,
                    enable_crystal_skin=enable_crystal_skin,
                    crystal_skin_strength=crystal_skin_strength,
                    enable_glossy_lips=enable_glossy_lips,
                    lip_gloss=lip_gloss,
                    lip_vibrance=lip_vibrance,
                    enable_doll_eye=enable_doll_eye,
                    doll_eye_depth=doll_eye_depth,
                    enable_golden_hour=enable_golden_hour,
                    golden_warmth=golden_warmth,
                    golden_bloom=golden_bloom,
                    enable_super_clarity=enable_super_clarity,
                    clarity_strength=clarity_strength,
                    enable_deblur=enable_deblur,
                    deblur_strength=deblur_strength,
                    enable_dehaze=enable_dehaze,
                    dehaze_strength=dehaze_strength,
                    sharpen_amount=sharpen_amount
                )



            
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
                try:
                    inv_restored = cv2.warpAffine(restored_face_up, inv_aff, (w_up, h_up))
                except Exception:
                    inv_restored = None
            else:
                # Blend with original cropped face to preserve original high-resolution details when w > 0
                if w > 0.0:
                    restored_face = cv2.addWeighted(cropped_face, w, restored_face, 1.0 - w, 0.0)
                
                # Add an offset to inverse affine matrix, for more precise back alignment
                extra_offset = 0
                inv_aff[:, 2] += extra_offset
                face_size = raw_face_size
                try:
                    inv_restored = cv2.warpAffine(restored_face, inv_aff, (w_up, h_up))
                except Exception:
                    inv_restored = None
            
            if inv_restored is None:
                continue

            # Create boundary mask
            mask = np.ones(face_size, dtype=np.float32)
            try:
                inv_mask = cv2.warpAffine(mask, inv_aff, (w_up, h_up))
            except Exception:
                continue
            
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
            
        # Free intermediate memory allocations
        gc.collect()
        return upsample_img

    def process_video(
        self,
        input_video_path: str,
        output_video_path: str,
        w: float = 0.5,
        detection_model: str = 'retinaface_mobile0.25',
        upscale: int = 1,
        frame_stride: int = 1,
        max_frames: int = None,
        progress_callback: callable = None,
        **enhancer_kwargs
    ) -> dict:
        """
        Process a video file frame-by-frame with face detection and AI restoration.
        Preserves video frame rate and attempts audio re-attachment via ffmpeg.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None and max_frames > 0:
            total_frames = min(total_frames, max_frames)
        
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w = orig_w * upscale
        out_h = orig_h * upscale

        temp_out = output_video_path + ".temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_out, fourcc, fps, (out_w, out_h))

        t0 = time.time()
        frame_idx = 0
        faces_restored_total = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames is not None and frame_idx >= max_frames):
                    break

                if frame_idx % frame_stride == 0:
                    enhanced_frame = self.process_image(
                        frame,
                        w=w,
                        detection_model=detection_model,
                        upscale=upscale,
                        **enhancer_kwargs
                    )
                else:
                    if upscale != 1:
                        enhanced_frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        enhanced_frame = frame

                writer.write(enhanced_frame)
                frame_idx += 1

                if progress_callback:
                    pct = min(1.0, frame_idx / max(1, total_frames))
                    progress_callback(
                        "video_render",
                        pct,
                        f"Rendering frame {frame_idx}/{total_frames} ({pct:.1%})"
                    )

        finally:
            cap.release()
            writer.release()

        elapsed = time.time() - t0

        # Attempt to copy audio from original video using ffmpeg
        audio_copied = False
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_out,
                "-i", input_video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                output_video_path
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=30)
            if res.returncode == 0 and os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                audio_copied = True
                try:
                    os.remove(temp_out)
                except OSError:
                    pass
        except Exception:
            audio_copied = False

        if not audio_copied:
            if os.path.exists(output_video_path):
                try:
                    os.remove(output_video_path)
                except OSError:
                    pass
            os.rename(temp_out, output_video_path)

        return {
            "total_frames": frame_idx,
            "fps": fps,
            "duration_sec": elapsed,
            "avg_fps": frame_idx / max(0.001, elapsed),
            "output_path": output_video_path,
            "audio_preserved": audio_copied
        }

    def process_batch_images(
        self,
        images_dict: dict,
        progress_callback: callable = None,
        **process_kwargs
    ) -> dict:
        """
        Process a batch dictionary of {filename: cv2_image_bgr}.
        Returns dictionary of results with quality report metrics.
        """
        total = len(images_dict)
        results = {}
        t_start = time.time()

        for idx, (filename, img_bgr) in enumerate(images_dict.items()):
            t0 = time.time()
            enhanced = self.process_image(img_bgr, **process_kwargs)
            dur = time.time() - t0

            q_report = {}
            if hasattr(self, 'wink_enhancer'):
                q_report = self.wink_enhancer.calculate_quality_report(img_bgr, enhanced)

            results[filename] = {
                "orig": img_bgr,
                "enhanced": enhanced,
                "report": q_report,
                "duration": dur
            }

            if progress_callback:
                pct = (idx + 1) / total
                progress_callback("batch_progress", pct, f"Processed {idx + 1}/{total}: {filename} ({dur:.2f}s)")

        total_duration = time.time() - t_start
        return {
            "items": results,
            "total_count": total,
            "total_duration": total_duration,
            "avg_time_per_image": total_duration / max(1, total)
        }

    def generate_html_report(self, batch_data: dict) -> str:
        """
        Generate a self-contained, beautifully styled HTML Quality Report Card with embedded base64 thumbnails.
        """
        import base64
        items = batch_data.get("items", {})
        total_count = batch_data.get("total_count", len(items))
        total_dur = batch_data.get("total_duration", 0.0)

        # Compute summary stats
        sharp_gains = [it["report"].get("sharpness_gain_pct", 0) for it in items.values() if "report" in it]
        avg_gain = sum(sharp_gains) / max(1, len(sharp_gains))

        rows_html = ""
        for name, data in items.items():
            orig = data["orig"]
            enh = data["enhanced"]
            rep = data.get("report", {})
            dur = data.get("duration", 0.0)

            # Generate small thumbnails
            thumb_h = 160
            scale_o = thumb_h / max(1, orig.shape[0])
            thumb_o = cv2.resize(orig, (int(orig.shape[1] * scale_o), thumb_h), interpolation=cv2.INTER_AREA)
            scale_e = thumb_h / max(1, enh.shape[0])
            thumb_e = cv2.resize(enh, (int(enh.shape[1] * scale_e), thumb_h), interpolation=cv2.INTER_AREA)

            _, buf_o = cv2.imencode('.jpg', thumb_o, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _, buf_e = cv2.imencode('.jpg', thumb_e, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_o = base64.b64encode(buf_o).decode('utf-8')
            b64_e = base64.b64encode(buf_e).decode('utf-8')

            rows_html += f"""
            <tr>
                <td style="font-weight: 600; color: #f3f0ff;">{name}</td>
                <td><img src="data:image/jpeg;base64,{b64_o}" style="border-radius: 8px; max-height: 120px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);"/></td>
                <td><img src="data:image/jpeg;base64,{b64_e}" style="border-radius: 8px; max-height: 120px; box-shadow: 0 4px 12px rgba(124,58,237,0.3);"/></td>
                <td style="color: #34d399; font-weight: 700; font-size: 1.1rem;">+{rep.get('sharpness_gain_pct', 0)}%</td>
                <td style="color: #60a5fa; font-weight: 600;">{rep.get('tone_fidelity_pct', 0)}%</td>
                <td style="color: #c4b5fd;">{dur:.2f}s</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>AI Portrait Enhancement - Quality Scorecard Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #090714; color: #f3f0ff; padding: 40px 20px; }}
        .card {{ max-width: 1000px; margin: 0 auto; background: #130f28; border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 32px; box-shadow: 0 12px 36px rgba(0,0,0,0.6); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .title {{ font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .subtitle {{ color: #94a3b8; font-size: 0.95rem; margin-top: 6px; }}
        .metrics-bar {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 30px; }}
        .metric-box {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; text-align: center; }}
        .metric-title {{ font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px; }}
        .metric-val {{ font-size: 1.5rem; font-weight: 800; color: #fff; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: rgba(255,255,255,0.06); color: #c4b5fd; text-align: left; padding: 12px 16px; font-size: 0.85rem; text-transform: uppercase; border-radius: 6px; }}
        td {{ padding: 14px 16px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.95rem; vertical-align: middle; }}
        tr:hover {{ background: rgba(255,255,255,0.02); }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div class="title">✨ AI Portrait Restoration Report</div>
            <div class="subtitle">Wink Studio Pro • Batch Execution Quality Scorecard</div>
        </div>
        <div class="metrics-bar">
            <div class="metric-box"><div class="metric-title">Total Portraits</div><div class="metric-val">{total_count}</div></div>
            <div class="metric-box"><div class="metric-title">Avg Sharpness Gain</div><div class="metric-val" style="color: #34d399;">+{avg_gain:.1f}%</div></div>
            <div class="metric-box"><div class="metric-title">Total Duration</div><div class="metric-val">{total_dur:.2f}s</div></div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Filename</th>
                    <th>Original</th>
                    <th>Wink Enhanced HD</th>
                    <th>Sharpness</th>
                    <th>Skin Tone</th>
                    <th>Speed</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>"""
        return html
