---
name: noise-estimation-denoising
description: Non-Local Means (NLM) denoising, wavelet thresholding, and bilateral filtering for noise estimation and suppression prior to super-resolution.
---

# Noise Estimation & Pre-Denoising Skill

## Overview
High-frequency noise in input images gets amplified during AI super-resolution and face restoration. Pre-denoising with Fast Non-Local Means or Bilateral filtering smooths uniform noise while preserving structural edges.

## Key Methods
1. **Noise Estimation**: Compute standard deviation of Laplacian response ($\sigma_n = 	ext{std}(
abla^2 I)$).
2. **Fast NLM Denoising**: `cv2.fastNlMeansDenoisingColored(img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)`.
3. **Bilateral Filter**: `cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)`.
