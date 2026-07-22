---
name: eye-and-teeth-enhancement
description: Facial parsing targeted enhancements for boosting iris clarity, catchlight sparkle, ocular contrast, and natural teeth brightening without over-bleaching.
---

# Eye & Teeth Targeted Enhancement Skill

## Overview
Eyes and teeth are central focal points in portrait photography. This skill targets parsed eye and mouth masks to boost iris sharpness, add catchlight clarity, and gently brighten teeth while maintaining natural skin balance.

## Procedures
1. **Iris & Ocular Boost**:
   - Extract eye mask from face parsing output.
   - Apply localized CLAHE and high-pass sharpening on the ocular region ($1.2	imes$ scale).
2. **Teeth Brightening**:
   - Isolate inner mouth/teeth mask.
   - Convert to HSV space; slightly desaturate yellow ($S 	imes 0.85$) and boost luminance ($V 	imes 1.08$).

## Code Example
```python
import cv2
import numpy as np

def enhance_eyes_and_teeth(img_bgr, eye_mask, teeth_mask):
    out = img_bgr.copy()
    
    # Eye sharpness boost
    if np.any(eye_mask > 0):
        eyes_sharp = cv2.addWeighted(img_bgr, 1.3, cv2.GaussianBlur(img_bgr, (0,0), 2.0), -0.3, 0)
        mask_eye_f = (eye_mask.astype(np.float32) / 255.0)[:, :, None]
        out = np.uint8(out * (1 - mask_eye_f) + eyes_sharp * mask_eye_f)
        
    # Teeth whitening in HSV
    if np.any(teeth_mask > 0):
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= np.where(teeth_mask > 0, 0.85, 1.0) # desaturate yellow
        hsv[:, :, 2] *= np.where(teeth_mask > 0, 1.08, 1.0) # boost brightness
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    return out
```
