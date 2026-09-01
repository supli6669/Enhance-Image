import cv2
import numpy as np

class WinkQualityEnhancer:
    """
    High-speed OpenCV/NumPy post-processor for Wink-level visual enhancement:
    1. Real Skin Grain & Texture (Frequency Separation)
    2. Localized Eye & Lip Sharpening / Sparkle Boost
    3. LAB CLAHE Lighting & Micro-Contrast Tone Balance
    """
    def __init__(self):
        self.clahe_eye = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.clahe_lab = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def correct_chromatic_aberration(self, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Correct lateral chromatic aberration (color fringing) on lens edges.
        Uses radial channel realignment of Red and Blue relative to Green center.
        """
        if strength <= 0.0 or img is None:
            return img
            
        try:
            h, w = img.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            
            # Split channels (BGR)
            b, g, r = cv2.split(img)
            
            # Radial scale factors for B and R relative to G center
            scale_b = 1.0 + (0.0015 * strength)
            scale_r = 1.0 - (0.0015 * strength)
            
            # Affine matrix for scaling around image center
            mat_b = cv2.getRotationMatrix2D((cx, cy), 0, scale_b)
            mat_r = cv2.getRotationMatrix2D((cx, cy), 0, scale_r)
            
            b_aligned = cv2.warpAffine(b, mat_b, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            r_aligned = cv2.warpAffine(r, mat_r, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            
            corrected = cv2.merge([b_aligned, g, r_aligned])
            return corrected
        except Exception as e:
            print(f"[WinkEnhancer] Chromatic aberration correction warning: {e}")
            return img

    def apply_skin_grain(self, restored_face: np.ndarray, cropped_original: np.ndarray, skin_mask: np.ndarray = None, grain_amount: float = 0.15, skin_soften: float = 0.3) -> np.ndarray:
        """
        Pro Studio Skin Texture Synthesis 2.0:
        Extract adaptive high-pass frequency texture from cropped_original and inject into restored_face
        combined with parsing-guided bilateral skin tone softening to eliminate plastic/soapy skin look
        while preserving authentic pore structures and AI facial details.
        """
        if (grain_amount <= 0.0 and skin_soften <= 0.0) or cropped_original is None:
            return restored_face
            
        try:
            h, w = restored_face.shape[:2]
            # Ensure same dimensions
            if restored_face.shape[:2] != cropped_original.shape[:2]:
                cropped_orig_resized = cv2.resize(cropped_original, (w, h), interpolation=cv2.INTER_LANCZOS4)
            else:
                cropped_orig_resized = cropped_original

            # Adaptive dynamic kernel based on face crop resolution
            k_val = max(3, int(min(h, w) / 100) * 2 + 1)
            ksize = (k_val, k_val)

            # Frequency Separation: Extract high-frequency pore details from original
            orig_blur = cv2.GaussianBlur(cropped_orig_resized, ksize, 0)
            high_freq = cv2.subtract(cropped_orig_resized.astype(np.int16), orig_blur.astype(np.int16))
            grain_layer = (high_freq * grain_amount).clip(-128, 127)

            # Gentle edge-preserving bilateral skin softening for studio porcelain smoothness
            if skin_soften > 0.0:
                soft_base = cv2.bilateralFilter(restored_face, d=5, sigmaColor=int(25 * skin_soften), sigmaSpace=int(25 * skin_soften))
                base_face = cv2.addWeighted(restored_face, 1.0 - skin_soften * 0.5, soft_base, skin_soften * 0.5, 0)
            else:
                base_face = restored_face

            if skin_mask is not None:
                skin_mask_2d = np.squeeze(skin_mask)
                if skin_mask_2d.ndim == 2:
                    if skin_mask_2d.shape != (h, w):
                        skin_mask_resized = cv2.resize(skin_mask_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        skin_mask_resized = skin_mask_2d
                    
                    # Skin category in facexlib parse mask is index 1
                    skin_binary = (skin_mask_resized == 1).astype(np.float32)
                    # Smooth mask edge for seamless alpha blending
                    skin_binary = cv2.GaussianBlur(skin_binary, (5, 5), 0)[:, :, np.newaxis]
                    
                    # Apply softened base only to skin region, keeping facial organs crisp
                    blended_base = (base_face.astype(np.float32) * skin_binary) + (restored_face.astype(np.float32) * (1.0 - skin_binary))
                    blended = blended_base + grain_layer * skin_binary
                else:
                    blended = base_face.astype(np.float32) + grain_layer
            else:
                blended = base_face.astype(np.float32) + grain_layer

            return np.clip(blended, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[WinkEnhancer] Skin grain warning: {e}")
            return restored_face

    def whiten_teeth(self, face_img: np.ndarray, parse_mask: np.ndarray = None, strength: float = 0.35) -> np.ndarray:
        """
        Studio Natural Teeth Whitening:
        Isolates teeth/mouth region using parsing mask, desaturates yellow color cast in LAB space,
        and lifts luminance naturally without artificial chalky/over-bleached artifacts.
        """
        if strength <= 0.0 or parse_mask is None or face_img is None:
            return face_img

        try:
            parse_mask_2d = np.squeeze(parse_mask)
            if parse_mask_2d.ndim != 2:
                return face_img

            h, w = face_img.shape[:2]
            if parse_mask_2d.shape[:2] != (h, w):
                parse_mask_res = cv2.resize(parse_mask_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                parse_mask_res = parse_mask_2d

            # Teeth / Mouth opening mask (index 11 in facexlib / BiSeNet)
            teeth_mask = (parse_mask_res == 11).astype(np.uint8)
            if not np.any(teeth_mask):
                return face_img

            # Convert to LAB color space
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB).astype(np.float32)
            l, a, b = cv2.split(lab)

            # Mask smoothing for seamless transition
            teeth_mask_float = cv2.GaussianBlur(teeth_mask.astype(np.float32), (5, 5), 0)

            # In LAB: b > 128 is yellow, b < 128 is blue. Desaturate yellow cast:
            b_diff = b - 128.0
            b_corrected = b - (b_diff * strength * 0.70)
            
            # Gently lift lightness L on teeth (boost bright teeth pixels safely)
            l_boost = l + ((255.0 - l) * strength * 0.18)
            
            l_final = l * (1.0 - teeth_mask_float) + l_boost * teeth_mask_float
            b_final = b * (1.0 - teeth_mask_float) + b_corrected * teeth_mask_float

            lab_whitened = cv2.merge([l_final, a, b_final])
            lab_whitened = np.clip(lab_whitened, 0, 255).astype(np.uint8)
            return cv2.cvtColor(lab_whitened, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"[WinkEnhancer] Teeth whitening warning: {e}")
            return face_img

    def brighten_eyes_and_sclera(self, face_img: np.ndarray, parse_mask: np.ndarray = None, strength: float = 0.35) -> np.ndarray:
        """
        Studio Ocular Catchlight & Sclera Glow:
        Boosts iris specular highlights (catchlights) and clarifies eye whites without over-sharpening.
        """
        if strength <= 0.0 or parse_mask is None or face_img is None:
            return face_img

        try:
            parse_mask_2d = np.squeeze(parse_mask)
            if parse_mask_2d.ndim != 2:
                return face_img

            h, w = face_img.shape[:2]
            if parse_mask_2d.shape[:2] != (h, w):
                parse_mask_res = cv2.resize(parse_mask_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                parse_mask_res = parse_mask_2d

            eye_mask = ((parse_mask_res == 4) | (parse_mask_res == 5)).astype(np.uint8)
            if not np.any(eye_mask):
                return face_img

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eye_mask_dilated = cv2.dilate(eye_mask, kernel, iterations=1)
            eye_mask_float = cv2.GaussianBlur(eye_mask_dilated.astype(np.float32), (3, 3), 0)[:, :, np.newaxis]

            # In LAB color space: apply CLAHE for iris depth and specular highlights
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe_eye.apply(l)

            # Specular catchlight amplification on bright reflections
            bright_catchlight = (l > 190).astype(np.float32)
            l_sparkle = np.clip(l.astype(np.float32) + bright_catchlight * (35.0 * strength), 0, 255).astype(np.uint8)
            l_enhanced = np.where(eye_mask_dilated == 1, cv2.addWeighted(l_clahe, 0.7, l_sparkle, 0.3, 0), l)

            lab_eye = cv2.merge([l_enhanced, a, b])
            eye_bgr = cv2.cvtColor(lab_eye, cv2.COLOR_LAB2BGR)

            # High-pass subtle micro-sharpening on eye details (eyelashes, pupils)
            eye_blur = cv2.GaussianBlur(eye_bgr, (3, 3), 0)
            eye_sharp = cv2.addWeighted(eye_bgr, 1.0 + (0.35 * strength), eye_blur, -(0.35 * strength), 0)

            result = (face_img.astype(np.float32) * (1.0 - eye_mask_float) + eye_sharp.astype(np.float32) * eye_mask_float)
            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[WinkEnhancer] Eye brightness warning: {e}")
            return face_img

    def balance_portrait_lighting_and_tone(self, face_img: np.ndarray, strength: float = 0.25) -> np.ndarray:
        """
        Studio Lighting & Skin Glow Tone Balancer:
        Performs Gray-World auto white balance and shadows recovery with healthy skin radiance.
        """
        if strength <= 0.0 or face_img is None:
            return face_img

        try:
            # 1. Shades of Gray Auto White Balance
            img_f = face_img.astype(np.float32)
            mean_b = np.mean(img_f[:, :, 0]) + 1e-5
            mean_g = np.mean(img_f[:, :, 1]) + 1e-5
            mean_r = np.mean(img_f[:, :, 2]) + 1e-5
            mean_gray = (mean_b + mean_g + mean_r) / 3.0

            scale_b = 1.0 + ((mean_gray / mean_b) - 1.0) * (strength * 0.6)
            scale_g = 1.0 + ((mean_gray / mean_g) - 1.0) * (strength * 0.6)
            scale_r = 1.0 + ((mean_gray / mean_r) - 1.0) * (strength * 0.6)

            wb_img = np.zeros_like(img_f)
            wb_img[:, :, 0] = img_f[:, :, 0] * scale_b
            wb_img[:, :, 1] = img_f[:, :, 1] * scale_g
            wb_img[:, :, 2] = img_f[:, :, 2] * scale_r
            wb_img = np.clip(wb_img, 0, 255).astype(np.uint8)

            # 2. LAB Shadow Lift & Radiance Glow
            lab = cv2.cvtColor(wb_img, cv2.COLOR_BGR2LAB).astype(np.float32)
            l, a, b = cv2.split(lab)

            # Quadratic shadow recovery curve: lifts dark shadows while protecting highlights
            shadow_boost = (l * (255.0 - l) / 255.0) * (strength * 0.25)
            l_lifted = np.clip(l + shadow_boost, 0, 255)

            # Subtle pink/warm skin tone enrichment (slight a-channel boost)
            a_glow = np.clip(a + (strength * 2.0), 0, 255)

            balanced_lab = cv2.merge([l_lifted, a_glow, b])
            balanced_bgr = cv2.cvtColor(balanced_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            return cv2.addWeighted(face_img, 1.0 - strength, balanced_bgr, strength, 0)
        except Exception as e:
            print(f"[WinkEnhancer] Portrait lighting balance warning: {e}")
            return face_img

    def enhance_eyes_and_lips(self, face_img: np.ndarray, parse_mask: np.ndarray = None, enable_eyes: bool = True, enable_lips: bool = True, enable_teeth: bool = True, enable_tone_glow: bool = True) -> np.ndarray:
        """
        Enhance eyes (catchlight sparkle), teeth (whitening), lips and overall studio skin glow.
        """
        result = face_img.copy()

        # 1. Studio Lighting and Skin Glow Tone Balance
        if enable_tone_glow:
            result = self.balance_portrait_lighting_and_tone(result, strength=0.25)

        # 2. Teeth Whitening
        if enable_teeth and parse_mask is not None:
            result = self.whiten_teeth(result, parse_mask, strength=0.35)

        # 3. Ocular Catchlight & Sclera Glow
        if enable_eyes and parse_mask is not None:
            result = self.brighten_eyes_and_sclera(result, parse_mask, strength=0.35)

        if parse_mask is None:
            # Fallback: General soft unsharp mask on entire face
            blur = cv2.GaussianBlur(result, (0, 0), 2.0)
            return cv2.addWeighted(result, 1.15, blur, -0.15, 0)

        try:
            parse_mask_2d = np.squeeze(parse_mask)
            if parse_mask_2d.ndim != 2:
                blur = cv2.GaussianBlur(result, (0, 0), 2.0)
                return cv2.addWeighted(result, 1.15, blur, -0.15, 0)

            h, w = result.shape[:2]
            if parse_mask_2d.shape[:2] != (h, w):
                parse_mask_res = cv2.resize(parse_mask_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                parse_mask_res = parse_mask_2d

            # Facial feature mask IDs in facexlib:
            # 12: Upper Lip, 13: Lower Lip
            lip_mask = ((parse_mask_res == 12) | (parse_mask_res == 13)).astype(np.uint8)

            # 4. Enhance Lips: Subtle natural saturation & contrast boost
            if enable_lips and np.any(lip_mask):
                lip_mask_float = cv2.GaussianBlur(lip_mask.astype(np.float32), (3, 3), 0)[:, :, np.newaxis]
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
                h_ch, s_ch, v_ch = cv2.split(hsv)
                s_boost = np.clip(s_ch * 1.15, 0, 255)
                v_boost = np.clip(v_ch * 1.05, 0, 255)
                hsv_lip = cv2.merge([h_ch, s_boost, v_boost]).astype(np.uint8)
                lip_bgr = cv2.cvtColor(hsv_lip, cv2.COLOR_HSV2BGR)
                result = (result.astype(np.float32) * (1.0 - lip_mask_float) + lip_bgr.astype(np.float32) * lip_mask_float).astype(np.uint8)

            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[WinkEnhancer] Organ enhancement warning: {e}")
            return result

    def balance_skin_tone_lab(self, face_img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on the L channel of LAB space to balance skin lighting, micro-contrast, and dynamic range.
        """
        try:
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe_lab.apply(l)
            # Soft blend to avoid over-exposure
            l_final = cv2.addWeighted(l, 0.6, l_clahe, 0.4, 0)
            lab_balanced = cv2.merge([l_final, a, b])
            return cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"[WinkEnhancer] LAB tone balance warning: {e}")
            return face_img

    def match_color_reinhard(self, target_img: np.ndarray, source_img: np.ndarray, blend: float = 0.5) -> np.ndarray:
        """
        Reinhard Color Transfer: Match color statistics (mean and std dev in LAB space)
        of target_img (restored AI face) to source_img (original cropped face/neck).
        """
        if source_img is None or blend <= 0.0:
            return target_img

        try:
            if target_img.shape[:2] != source_img.shape[:2]:
                source_res = cv2.resize(source_img, (target_img.shape[1], target_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            else:
                source_res = source_img

            target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)
            source_lab = cv2.cvtColor(source_res, cv2.COLOR_BGR2LAB).astype(np.float32)

            t_mean, t_std = cv2.meanStdDev(target_lab)
            s_mean, s_std = cv2.meanStdDev(source_lab)

            t_mean = t_mean.flatten()
            t_std = np.maximum(t_std.flatten(), 1e-5)
            s_mean = s_mean.flatten()
            s_std = s_std.flatten()

            res_lab = np.zeros_like(target_lab)
            for i in range(3):
                res_lab[:, :, i] = ((target_lab[:, :, i] - t_mean[i]) * (s_std[i] / t_std[i])) + s_mean[i]

            res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
            matched_bgr = cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

            return cv2.addWeighted(target_img, 1.0 - blend, matched_bgr, blend, 0)
        except Exception as e:
            print(f"[WinkEnhancer] Color match warning: {e}")
            return target_img

    def apply_adaptive_sharpening(self, img: np.ndarray, sharpen_amount: float = 0.2) -> np.ndarray:
        """
        Multi-Scale Edge-Aware Sharpening:
        Extracts structural edge mask using Sobel magnitude and applies dual-scale
        Unsharp Masking (fine micro-details + coarse structural edges) without halos.
        """
        if sharpen_amount <= 0.0 or img is None:
            return img

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Sobel edge magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = cv2.magnitude(grad_x, grad_y)
            edge_norm = cv2.normalize(edge_mag, None, 0.0, 1.0, cv2.NORM_MINMAX)[:, :, np.newaxis]
            
            # Dual-scale Unsharp Masking
            blur_fine = cv2.GaussianBlur(img, (3, 3), 1.0)
            blur_coarse = cv2.GaussianBlur(img, (7, 7), 3.0)
            
            sharp_fine = cv2.addWeighted(img, 1.0 + sharpen_amount, blur_fine, -sharpen_amount, 0)
            sharp_coarse = cv2.addWeighted(img, 1.0 + (sharpen_amount * 0.5), blur_coarse, -(sharpen_amount * 0.5), 0)
            
            # Blend sharp layers weighted by edge mask
            out = img.astype(np.float32) * (1.0 - edge_norm) + (sharp_fine.astype(np.float32) * 0.7 + sharp_coarse.astype(np.float32) * 0.3) * edge_norm
            return np.clip(out, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[WinkEnhancer] Adaptive sharpening warning: {e}")
            return img

    def enhance_face(self, restored_face: np.ndarray, cropped_original: np.ndarray = None, parse_mask: np.ndarray = None, wink_mode: bool = True, eye_enhancement: bool = True, skin_grain: float = 0.15, color_match: bool = True, enable_eyes: bool = True, enable_lips: bool = True, enable_skin: bool = True, enable_teeth: bool = True, enable_tone_glow: bool = True, sharpen_amount: float = 0.2) -> np.ndarray:
        """
        Master method to execute Wink-level enhancement pipeline on a restored face crop.
        """
        if not wink_mode:
            return restored_face

        out_face = restored_face.copy()

        # Step A: Studio Lighting and Skin Glow Tone Balance (Auto White Balance & Radiance)
        if enable_tone_glow:
            out_face = self.balance_portrait_lighting_and_tone(out_face, strength=0.25)

        # Step B: Reinhard Color Transfer (Auto Skin Tone Alignment to original face/neck)
        if color_match and cropped_original is not None:
            out_face = self.match_color_reinhard(out_face, cropped_original, blend=0.4)

        # Step C: Skin tone & micro-contrast balance
        out_face = self.balance_skin_tone_lab(out_face)

        # Step D: Eye (sparkle), Teeth (whitening) & Lip local enhancement
        if eye_enhancement and (enable_eyes or enable_lips or enable_teeth):
            out_face = self.enhance_eyes_and_lips(
                out_face,
                parse_mask=parse_mask,
                enable_eyes=enable_eyes,
                enable_lips=enable_lips,
                enable_teeth=enable_teeth,
                enable_tone_glow=False
            )

        # Step E: Multi-Scale Edge-Aware Adaptive Sharpening
        if sharpen_amount > 0.0:
            out_face = self.apply_adaptive_sharpening(out_face, sharpen_amount=sharpen_amount)

        # Step F: Real Skin Grain Injection (Frequency Separation)
        if enable_skin and skin_grain > 0.0 and cropped_original is not None:
            out_face = self.apply_skin_grain(out_face, cropped_original, skin_mask=parse_mask, grain_amount=skin_grain)

        return out_face


    def calculate_sharpness(self, img: np.ndarray) -> float:
        """Calculate image sharpness using Variance of Laplacian."""
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def calculate_quality_report(self, orig_img: np.ndarray, enhanced_img: np.ndarray, face_count: int = 0) -> dict:
        """
        Generate AI Quality Score & Comparison metrics report.
        """
        orig_sharpness = self.calculate_sharpness(orig_img)
        enh_sharpness = self.calculate_sharpness(enhanced_img)
        
        sharpness_gain_pct = ((enh_sharpness - orig_sharpness) / max(orig_sharpness, 1e-5)) * 100.0
        sharpness_gain_pct = float(np.clip(sharpness_gain_pct, 0.0, 1000.0))

        # Skin tone fidelity score (using LAB luminance correlation)
        try:
            o_res = cv2.resize(orig_img, (enhanced_img.shape[1], enhanced_img.shape[0]))
            o_lab = cv2.cvtColor(o_res, cv2.COLOR_BGR2LAB).astype(np.float32)
            e_lab = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2LAB).astype(np.float32)
            diff = np.mean(np.abs(o_lab[:, :, 1:] - e_lab[:, :, 1:]))
            tone_fidelity_pct = float(np.clip(100.0 - (diff * 1.5), 70.0, 99.9))
        except Exception:
            tone_fidelity_pct = 95.0

        return {
            'orig_sharpness': round(orig_sharpness, 1),
            'enh_sharpness': round(enh_sharpness, 1),
            'sharpness_gain_pct': round(sharpness_gain_pct, 1),
            'face_count': face_count,
            'tone_fidelity_pct': round(tone_fidelity_pct, 1)
        }

    def create_comparison_animation(self, orig_img: np.ndarray, enhanced_img: np.ndarray, num_frames: int = 24, fps: int = 12, max_dim: int = 512) -> bytes:
        """
        Create a smooth animated GIF sliding back and forth between Original and Enhanced image.
        """
        import io
        from PIL import Image, ImageDraw

        if orig_img is None or enhanced_img is None:
            return b""

        # Resize to matching dimensions with max_dim cap for fast encoding and lightweight size
        h, w = enhanced_img.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        new_w, new_h = max(64, int(w * scale)), max(64, int(h * scale))
        
        orig_res = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        enh_res = cv2.resize(enhanced_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        orig_rgb = cv2.cvtColor(orig_res, cv2.COLOR_BGR2RGB)
        enh_rgb = cv2.cvtColor(enh_res, cv2.COLOR_BGR2RGB)
        
        frames = []
        # Generate smooth ping-pong sweep: 0.05 -> 0.95 -> 0.05
        half_n = max(4, num_frames // 2)
        positions = np.linspace(0.05, 0.95, half_n)
        sweep_positions = list(positions) + list(reversed(positions))
        
        for pos in sweep_positions:
            split_x = int(new_w * pos)
            frame_np = np.zeros_like(orig_rgb)
            frame_np[:, :split_x] = orig_rgb[:, :split_x]
            frame_np[:, split_x:] = enh_rgb[:, split_x:]
            
            # Draw sleek vertical divider line
            line_w = max(2, int(new_w * 0.006))
            x1 = max(0, split_x - line_w // 2)
            x2 = min(new_w, split_x + line_w // 2 + 1)
            frame_np[:, x1:x2] = (255, 255, 255)
            
            pil_frame = Image.fromarray(frame_np)
            draw = ImageDraw.Draw(pil_frame)
            
            # Sleek corner watermark badges
            draw.rectangle([(8, 8), (78, 26)], fill=(20, 20, 20, 180))
            draw.text((12, 11), "ORIGINAL", fill=(255, 255, 255))
            
            draw.rectangle([(new_w - 88, 8), (new_w - 8, 26)], fill=(20, 20, 20, 180))
            draw.text((new_w - 82, 11), "RESTORED", fill=(0, 240, 255))
            
            frames.append(pil_frame)
            
        buf = io.BytesIO()
        duration_ms = int(1000 / max(1, fps))
        frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )
        return buf.getvalue()



