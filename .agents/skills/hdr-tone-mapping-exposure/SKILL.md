---
name: hdr-tone-mapping-exposure
description: High Dynamic Range tone mapping, shadow detail enhancement, highlight clipping control, gamma adjustments, and local tone mapping.
---

# HDR Tone Mapping & Exposure Recovery Skill

## Overview
Under-exposed shadows and over-exposed highlights lose texture details. Local tone mapping using Durand / Mantiuk tone mapping curves brings out hidden details in portrait shadows and backgrounds.

## Code Example
```python
import cv2
import numpy as np

def recover_shadows_and_highlights(img_bgr, gamma=1.1, shadow_boost=1.2):
    # Convert to float 0..1
    img_f = img_bgr.astype(np.float32) / 255.0
    
    # Shadow mask based on luminance
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow_mask = np.power(1.0 - gray, 2.0)[:, :, None]
    
    # Apply gamma & shadow boost
    corrected = np.power(img_f, 1.0 / gamma)
    boosted = corrected * (1.0 + (shadow_boost - 1.0) * shadow_mask)
    
    return np.clip(boosted * 255.0, 0, 255).astype(np.uint8)
```
