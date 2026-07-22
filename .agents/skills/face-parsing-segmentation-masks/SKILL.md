---
name: face-parsing-segmentation-masks
description: Facial component segmentation using ParseNet/BiSeNet, extracting eye, lip, skin, and hair masks for targeted post-processing and seamless edge blending.
---

# Face Parsing & Segmentation Masks Skill

## Overview
Face parsing segments a face image into semantic components: eyes, eyebrows, nose, mouth/lips, skin, hair, and background. Using these parsing masks allows targeted visual enhancements (e.g. skin grain injection, eye sharpening, lip tone balancing) without affecting non-target regions.

## Key Concepts
1. **Model Architecture**: BiSeNet or ParseNet trained on CelebAMask-HQ. Outputs 19 semantic channels.
2. **Mask Generation**:
   - `skin_mask = (parsing == 1)`
   - `eye_mask = (parsing == 4) | (parsing == 5)`
   - `mouth_mask = (parsing == 11) | (parsing == 12) | (parsing == 13)`
3. **Blending & Dilated Boundaries**:
   - Apply Gaussian blur (`cv2.GaussianBlur`) to mask edges for soft alpha transitions.
   - Use morphological erosion (`cv2.erode`) to avoid seam artifacts near hairline or jawline.

## Code Example
```python
import cv2
import numpy as np

def extract_parsing_masks(parsing_out):
    # parsing_out shape: (512, 512) containing class indices 0..18
    skin = np.uint8((parsing_out == 1) * 255)
    eyes = np.uint8(((parsing_out == 4) | (parsing_out == 5)) * 255)
    lips = np.uint8(((parsing_out == 11) | (parsing_out == 12) | (parsing_out == 13)) * 255)
    return skin, eyes, lips
```
