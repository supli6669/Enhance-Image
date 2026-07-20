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

    def apply_skin_grain(self, restored_face: np.ndarray, cropped_original: np.ndarray, skin_mask: np.ndarray = None, grain_amount: float = 0.15) -> np.ndarray:
        """
        Extract high-frequency texture from cropped_original and inject into restored_face
        to eliminate plastic/soapy skin look while preserving AI face restoration.
        """
        if grain_amount <= 0.0 or cropped_original is None:
            return restored_face
            
        try:
            # Ensure same dimensions
            if restored_face.shape[:2] != cropped_original.shape[:2]:
                cropped_orig_resized = cv2.resize(cropped_original, (restored_face.shape[1], restored_face.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            else:
                cropped_orig_resized = cropped_original

            # Frequency Separation: Extract high-frequency details from original
            orig_blur = cv2.GaussianBlur(cropped_orig_resized, (5, 5), 0)
            high_freq = cv2.subtract(cropped_orig_resized.astype(np.int16), orig_blur.astype(np.int16))
            
            # Scale high frequency grain
            grain_layer = (high_freq * grain_amount).clip(-128, 127)

            if skin_mask is not None:
                if skin_mask.shape[:2] != restored_face.shape[:2]:
                    skin_mask_resized = cv2.resize(skin_mask.astype(np.uint8), (restored_face.shape[1], restored_face.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    skin_mask_resized = skin_mask
                
                # Skin category in facexlib parse mask is index 1
                skin_binary = (skin_mask_resized == 1).astype(np.float32)[:, :, np.newaxis]
                # Smooth mask edge
                skin_binary = cv2.GaussianBlur(skin_binary, (5, 5), 0)
                if len(skin_binary.shape) == 2:
                    skin_binary = skin_binary[:, :, np.newaxis]
                
                blended = restored_face.astype(np.float32) + grain_layer * skin_binary
            else:
                blended = restored_face.astype(np.float32) + grain_layer

            return np.clip(blended, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[WinkEnhancer] Skin grain warning: {e}")
            return restored_face

    def enhance_eyes_and_lips(self, face_img: np.ndarray, parse_mask: np.ndarray = None) -> np.ndarray:
        """
        Enhance eyes (catchlight, contrast, sharpness) and lips using facial parsing mask.
        """
        if parse_mask is None:
            # Fallback: General soft unsharp mask on entire face
            blur = cv2.GaussianBlur(face_img, (0, 0), 2.0)
            return cv2.addWeighted(face_img, 1.15, blur, -0.15, 0)

        try:
            h, w = face_img.shape[:2]
            if parse_mask.shape[:2] != (h, w):
                parse_mask_res = cv2.resize(parse_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                parse_mask_res = parse_mask

            # Facial feature mask IDs in facexlib:
            # 4: Left Eye, 5: Right Eye, 6: Glasses, 11: Upper Lip, 12: Lower Lip, 13: Inner Mouth
            eye_mask = ((parse_mask_res == 4) | (parse_mask_res == 5) | (parse_mask_res == 6)).astype(np.uint8)
            lip_mask = ((parse_mask_res == 11) | (parse_mask_res == 12) | (parse_mask_res == 13)).astype(np.uint8)

            result = face_img.copy()

            # 1. Enhance Eyes: CLAHE on L channel + Unsharp Masking
            if np.any(eye_mask):
                # Expand eye mask slightly for seamless blending
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                eye_mask_dilated = cv2.dilate(eye_mask, kernel, iterations=1)
                
                # Convert to LAB for luminance contrast
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                l_eye_clahe = self.clahe_eye.apply(l)
                l_blended = np.where(eye_mask_dilated == 1, l_eye_clahe, l)
                lab_enhanced = cv2.merge([l_blended, a, b])
                result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
                
                # Unsharp Mask on Eyes
                eye_blur = cv2.GaussianBlur(result, (3, 3), 0)
                eye_sharp = cv2.addWeighted(result, 1.3, eye_blur, -0.3, 0)
                
                eye_mask_float = cv2.GaussianBlur(eye_mask_dilated.astype(np.float32), (3, 3), 0)[:, :, np.newaxis]
                result = (result * (1.0 - eye_mask_float) + eye_sharp * eye_mask_float).astype(np.uint8)

            # 2. Enhance Lips: Subtle contrast and saturation boost
            if np.any(lip_mask):
                lip_mask_float = cv2.GaussianBlur(lip_mask.astype(np.float32), (3, 3), 0)[:, :, np.newaxis]
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.where(lip_mask == 1, np.clip(hsv[:, :, 1] * 1.1, 0, 255), hsv[:, :, 1]) # Boost saturation slightly
                lip_enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                result = (result * (1.0 - lip_mask_float) + lip_enhanced * lip_mask_float).astype(np.uint8)

            return result
        except Exception as e:
            print(f"[WinkEnhancer] Eye/Lip enhancement warning: {e}")
            return face_img

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

    def enhance_face(self, restored_face: np.ndarray, cropped_original: np.ndarray = None, parse_mask: np.ndarray = None, wink_mode: bool = True, eye_enhancement: bool = True, skin_grain: float = 0.15) -> np.ndarray:
        """
        Master method to execute Wink-level enhancement pipeline on a restored face crop.
        """
        if not wink_mode:
            return restored_face

        out_face = restored_face.copy()

        # Step A: Skin tone & micro-contrast balance
        out_face = self.balance_skin_tone_lab(out_face)

        # Step B: Eye & Lip local enhancement
        if eye_enhancement:
            out_face = self.enhance_eyes_and_lips(out_face, parse_mask=parse_mask)

        # Step C: Real Skin Grain Injection (Frequency Separation)
        if skin_grain > 0.0 and cropped_original is not None:
            out_face = self.apply_skin_grain(out_face, cropped_original, skin_mask=parse_mask, grain_amount=skin_grain)

        return out_face
