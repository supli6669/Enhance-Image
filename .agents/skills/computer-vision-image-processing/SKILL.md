---
name: computer-vision-image-processing
description: Expert guidelines for computer vision pipelines, image restoration, blind degradation modeling, face landmark detection/alignment, parsing segmentation masks, color space transforms (BGR/RGB), affine matrix warping, and PSNR/SSIM/LPIPS evaluation metrics.
---

# Computer Vision & Image Processing Skill

Use this skill whenever working on image enhancement, face restoration, OpenCV transformations, landmark alignment, or image quality evaluation metrics.

## Core CV Guidelines & Best Practices

1. **Color Space & Channel Ordering Hygiene**:
   - OpenCV loads images in **BGR** format by default. PyTorch models (CodeFormer, Real-ESRGAN, VGG, ArcFace) expect **RGB** tensors normalized to $[0, 1]$ or $[-1, 1]$.
   - Always verify channel order before model inference:
     ```python
     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
     tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
     tensor = (tensor - 0.5) / 0.5  # normalize to [-1, 1] if required
     ```

2. **Landmark Detection & Align-Crop Pipeline**:
   - Face alignment requires detecting 5 facial landmarks (eye-left, eye-right, nose, mouth-left, mouth-right).
   - Compute similarity transformation matrix (affine matrix) to warp face crop to standard $512 \times 512$ canvas.
   - When pasting restored face crops back into the original image, invert the affine matrix (`cv2.invertAffTransform` or inverse matrix scaling):
     ```python
     if upscale > 1:
         inv_aff /= upscale
         inv_aff[:, 2] *= upscale
     inv_restored = cv2.warpAffine(restored_face, inv_aff, (target_w, target_h))
     ```

3. **Face Parsing Mask & Soft Edge Blending**:
   - Combine a Gaussian-blurred boundary erosion mask with the PyTorch face parsing segmentation mask (excluding background, hair, and neck artifacts) to prevent visual seam borders.
   - Always enforce contiguous memory layout (`np.ascontiguousarray`) before array slicing and elementwise operations to avoid stride mismatch errors.

4. **Blind Degradation Modeling**:
   - For real-world face restoration training, model complex compound degradations:
     - **Blur**: Gaussian blur ($\sigma \in [0.2, 3.0]$), generalized Gaussian blur, and motion blur.
     - **Noise**: Additive Gaussian noise + Poisson (shot) noise.
     - **Downsampling**: Random area, bilinear, or bicubic downsampling ($scale \in [1, 8]$).
     - **JPEG Artifacts**: Compression quality factor $Q \in [10, 80]$.

5. **Image Quality Assessment (IQA) Metrics**:
   - **Fidelity**: Measure Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
   - **Perceptual Quality**: Measure Learned Perceptual Image Patch Similarity (LPIPS) using VGG features.
   - **Distribution Realism**: Compute Fréchet Inception Distance (FID) over test dataset.
