---
name: unsharp-masking-deconvolution
description: Dynamic unsharp masking (USM), Richardson-Lucy non-blind deconvolution, and high-pass frequency boosting for crisp edge recovery.
---

# Unsharp Masking & Deconvolution Skill

## Overview
Restoring high-frequency details requires carefully tuned sharpening. Standard USM uses a Gaussian blur kernel to isolate high frequencies, while Richardson-Lucy deconvolution reverses point spread function (PSF) blurring.

## Formula & Code
USM Formula: $I_{sharp} = I + lpha (I - G_\sigma * I)$

```python
import cv2
import numpy as np

def adaptive_unsharp_mask(img, amount=0.5, radius=1.0, threshold=2):
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    diff = img.astype(np.float32) - blurred.astype(np.float32)
    mask = np.abs(diff) >= threshold
    sharpened = img.astype(np.float32) + amount * diff
    result = np.where(mask, sharpened, img.astype(np.float32))
    return np.clip(result, 0, 255).astype(np.uint8)
```
