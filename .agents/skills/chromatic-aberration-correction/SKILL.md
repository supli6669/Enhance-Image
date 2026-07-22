---
name: chromatic-aberration-correction
description: Detection and correction of lateral and longitudinal chromatic aberration (color fringing), radial channel alignment, and lens distortion repair.
---

# Chromatic Aberration & Color Fringing Correction Skill

## Overview
Chromatic aberration produces red/cyan or blue/yellow color fringes along high-contrast edges. Radial channel shift aligns Red and Blue channels to Green.

## Code Example
```python
import cv2
import numpy as np

def correct_chromatic_aberration(img_bgr):
    b, g, r = cv2.split(img_bgr)
    # Estimate warp matrix between channels using ECC or phase correlation
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERMCRITERIA_EPS | cv2.TERMCRITERIA_COUNT, 50, 1e-4)
    
    try:
        _, warp_matrix_r = cv2.findTransformECC(g, r, warp_matrix.copy(), warp_mode, criteria)
        r_corr = cv2.warpAffine(r, warp_matrix_r, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except cv2.error:
        r_corr = r
        
    return cv2.merge((b, g, r_corr))
```
