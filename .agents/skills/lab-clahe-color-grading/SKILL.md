---
name: lab-clahe-color-grading
description: LAB color space transformation, Contrast Limited Adaptive Histogram Equalization (CLAHE) on L-channel, and skin-tone preserving color enhancement.
---

# LAB CLAHE & Dynamic Color Grading Skill

## Overview
Restoring degraded images often leaves colors washed out or unevenly exposed. CLAHE (Contrast Limited Adaptive Histogram Equalization) applied in the LAB color space boosts local contrast and shadow detail without distorting human skin tones.

## Key Concepts
1. **LAB Color Space**: Separates Luminance ($L$) from color channels ($a$ for red-green, $b$ for yellow-blue).
2. **L-Channel CLAHE**: Enhances contrast locally on $L$ only, avoiding color shifts.
3. **Clip Limit & Tile Size**: Use `clipLimit=1.5` to `2.0` and `tileGridSize=(8,8)` for realistic enhancements.

## Implementation
```python
import cv2

def apply_lab_clahe(img_bgr, clip_limit=1.8, tile_size=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
```
