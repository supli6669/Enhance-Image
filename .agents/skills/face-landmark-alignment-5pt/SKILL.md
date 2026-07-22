---
name: face-landmark-alignment-5pt
description: Extracting 5-point facial landmarks (eyes, nose, mouth corners), computing similarity transformation matrices, $512 	imes 512$ alignment, and reverse warp-back.
---

# 5-Point Facial Landmark Alignment & Inverse Warp Skill

## Overview
Face restoration models require normalized face crops aligned to standard 5-point landmark canonical coordinates (left eye, right eye, nose tip, left mouth corner, right mouth corner). After restoration, the inverse affine transformation warps the crop back to the exact spatial location on the original frame.

## Canonical 512x512 Landmark Anchors
- Left Eye: $(192.98, 239.95)$
- Right Eye: $(318.55, 240.14)$
- Nose Tip: $(256.63, 314.02)$
- Left Mouth Corner: $(201.26, 371.41)$
- Right Mouth Corner: $(313.00, 371.19)$

## Code Snippet
```python
import cv2
import numpy as np

def get_similarity_transform(src_pts, dst_pts):
    # Computes 2x3 affine matrix for similarity transform (scale, rotation, translation)
    tfm = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
    return tfm
```
