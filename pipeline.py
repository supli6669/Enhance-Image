import os
import sys
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

# Ensure CodeFormer directory is on sys.path
project_dir = os.path.dirname(os.path.abspath(__file__))
codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")
if codeformer_dir not in sys.path:
    sys.path.insert(0, codeformer_dir)

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper

class LocalAIEnhancerPipeline:
    def __init__(self, device=None):
        """Initialize the CodeFormer model and helper pipeline."""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"[Pipeline] Initializing pipeline on device: {self.device}")
        
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
            raise FileNotFoundError(f"CodeFormer weights not found at {weights_path}. Please run download_weights.py first.")
            
        print(f"[Pipeline] Loading weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location=self.device)
        if 'params_ema' in checkpoint:
            self.net.load_state_dict(checkpoint['params_ema'])
        else:
            self.net.load_state_dict(checkpoint['params'])
        self.net.eval()
        print("[Pipeline] CodeFormer model loaded successfully.")

    def process_image(self, img, w=0.5, detection_model='retinaface_resnet50', upscale=2, blend_softness=0.5):
        """
        Enhance an image using the local CodeFormer pipeline.
        
        Args:
            img (numpy.ndarray): Input image in BGR format (OpenCV default).
            w (float): Fidelity weight (0.0 to 1.0). 0.0 for max quality, 1.0 for max fidelity.
            detection_model (str): Face detector model ('retinaface_resnet50', 'YOLOv5l', etc.).
            upscale (int): Upscale factor for output image.
            blend_softness (float): Blending mask softness (0.0 to 1.0).
            
        Returns:
            numpy.ndarray: Enhanced output image in BGR format.
        """
        # Set up FaceRestoreHelper
        # We tell it where the facelib weights are (under project weights/facelib)
        os.environ['FACE_DETECTOR_PATH'] = os.path.join(project_dir, "weights", "facelib")
        
        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext='png',
            use_parse=True,
            device=self.device
        )
        
        face_helper.clean_all()
        face_helper.read_image(img)
        
        # 1. Detect face landmarks and align/crop faces
        print(f"[Pipeline] Running face detection model: {detection_model}...")
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=False, 
            resize=640, 
            eye_dist_threshold=5
        )
        print(f"[Pipeline] Detected {num_det_faces} faces.")
        
        if num_det_faces == 0:
            # Return resized background if no faces are detected
            h, w_img, _ = img.shape
            return cv2.resize(img, (w_img * upscale, h * upscale), interpolation=cv2.INTER_LINEAR)
            
        face_helper.align_warp_face()
        
        # 2. Process each cropped face through CodeFormer
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            
            try:
                with torch.no_grad():
                    # Process with fidelity weight w
                    output = self.net(cropped_face_t, w=w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
            except Exception as error:
                print(f"[Pipeline] Failed CodeFormer inference for face index {idx}: {error}")
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
                
            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)
            
        # 3. Paste restored faces back into input image with custom soft blending
        print(f"[Pipeline] Seamlessly pasting {len(face_helper.restored_faces)} restored faces back...")
        face_helper.get_inverse_affine(None)
        
        enhanced_img = self.paste_faces_custom_blend(
            face_helper, 
            upscale=upscale, 
            blend_softness=blend_softness
        )
        return enhanced_img

    def paste_faces_custom_blend(self, face_helper, upscale, blend_softness):
        """Custom implementation of face pasting with adjustable soft blending mask."""
        h, w, _ = face_helper.input_img.shape
        h_up, w_up = int(h * upscale), int(w * upscale)
        
        # Initialize background image (upsampled background)
        upsample_img = cv2.resize(face_helper.input_img, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
        
        for restored_face, inverse_affine in zip(face_helper.restored_faces, face_helper.inverse_affine_matrices):
            # Alignment offset
            if upscale > 1:
                extra_offset = 0.5 * upscale
            else:
                extra_offset = 0
            
            # Create a local copy to avoid modifying original matrices
            inv_aff = inverse_affine.copy()
            inv_aff[:, 2] += extra_offset
            
            face_size = face_helper.face_size
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
                
                # Intersect soft boundary mask with face feature parsing mask
                fuse_mask = (inv_soft_parse_mask < inv_soft_mask).astype('int')
                inv_soft_mask = inv_soft_parse_mask * fuse_mask + inv_soft_mask * (1 - fuse_mask)
            
            # Merge restored face onto the background
            upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
            
        upsample_img = np.clip(upsample_img, 0, 255).astype(np.uint8)
        return upsample_img
