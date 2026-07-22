---
name: guided-filter-detail-enhancement
description: Guided image filtering, edge-preserving smoothing, detail layer amplification, and halo artifact suppression for ultra-sharp facial details without harsh noise.
---

# Guided Filter Detail Enhancement Skill

## Overview
Guided Filtering performs edge-preserving smoothing using a guidance image. By subtracting the guided-filtered base layer from the original image, we extract a pure edge-detail layer that can be amplified without causing ringing or halo artifacts typical of standard unsharp masking.

## Code Example
```python
import cv2
import numpy as np

def guided_detail_enhance(img, radius=4, eps=0.01, scale=1.4):
    # Guided filter edge-preserving smooth
    guided = cv2.ximgproc.guidedFilter(guide=img, src=img, radius=radius, eps=eps)
    detail = img.astype(np.float32) - guided.astype(np.float32)
    enhanced = img.astype(np.float32) + detail * (scale - 1.0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)
```
