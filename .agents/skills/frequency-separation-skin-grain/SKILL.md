---
name: frequency-separation-skin-grain
description: High/low frequency image decomposition, extracting authentic skin pore textures from original images, and blending texture back onto neural network face crops to eliminate plastic skin.
---

# Frequency Separation & Skin Grain Retention Skill

## Overview
Neural face restoration models (like CodeFormer) can produce over-smoothed, 'plastic' or 'soapy' skin textures. Frequency separation splits an image into Low Frequency (tones, color, shading) and High Frequency (fine pores, micro-textures, wrinkles). By injecting original high-frequency details onto the restored face crop, skin looks natural and hyper-realistic.

## Key Concepts
1. **Low Frequency Extraction**: Apply Gaussian blur (`sigma = 2.0 to 4.0`).
2. **High Frequency Extraction**: Subtract Low Frequency from original image: $HF = Original - LF + 128$.
3. **Texture Injection**: Blend original high frequency onto the restored face using parsing skin masks.

## Vectorized NumPy Implementation
```python
import cv2
import numpy as np

def inject_skin_grain(orig_crop, restored_crop, skin_mask, weight=0.15):
    # Convert to float32
    orig_f = orig_crop.astype(np.float32)
    rest_f = restored_crop.astype(np.float32)
    
    # Extract original high frequency
    lf_orig = cv2.GaussianBlur(orig_f, (0, 0), sigmaX=3.0)
    hf_orig = orig_f - lf_orig
    
    # Inject onto restored image inside skin mask
    blended = rest_f + hf_orig * weight
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Mask blend
    mask_3ch = (skin_mask.astype(np.float32) / 255.0)[:, :, None]
    result = np.uint8(restored_crop * (1 - mask_3ch) + blended * mask_3ch)
    return result
```
