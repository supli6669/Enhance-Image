---
name: super-resolution-tile-stitching
description: Tile-based image processing for ultra-high-resolution images under strict memory limits, including tile padding, overlap cosine blending, and seam removal.
---

# Super-Resolution Tile Processing & Stitching Skill

## Overview
High-resolution images (4K, 8K) exceed CPU/GPU RAM limits when passed whole through deep learning models. Tiling breaks images into overlapping patches, processes each patch independently, and stitches them with smooth cosine blending to eliminate visible seam borders.

## Key Concepts
1. **Tile Size & Overlap**: e.g., tile size $400 	imes 400$, overlap $40	ext{px}$.
2. **Padding**: Mirror pad outer borders to handle edge tiles seamlessly.
3. **Cosine Weight Mask**: Generate 2D Gaussian/cosine weight maps so patch overlap areas smoothly transition.

## Implementation Pattern
```python
import numpy as np

def generate_tile_weights(tile_h, tile_w, pad):
    # Smooth linear/cosine weight ramp at borders
    w_x = np.ones(tile_w, dtype=np.float32)
    w_y = np.ones(tile_h, dtype=np.float32)
    w_x[:pad] = np.linspace(0, 1, pad)
    w_x[-pad:] = np.linspace(1, 0, pad)
    w_y[:pad] = np.linspace(0, 1, pad)
    w_y[-pad:] = np.linspace(1, 0, pad)
    return np.outer(w_y, w_x)[:, :, None]
```
