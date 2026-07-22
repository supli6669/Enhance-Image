---
name: image-quality-metrics-eval
description: Quantitative and qualitative image evaluation using PSNR, SSIM, LPIPS perceptual distance, FID distribution score, NIQE no-reference quality, and ArcFace identity metrics.
---

# Image Quality Metrics & Evaluation Suite Skill

## Overview
Accurate evaluation of image restoration quality requires both full-reference metrics (comparing restored against ground truth) and no-reference quality metrics.

## Supported Metrics
1. **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel reconstruction fidelity ($dB$). Higher is better ($>28	ext{dB}$).
2. **SSIM (Structural Similarity Index)**: Measures structural/luminance closeness ($0..1$). Higher is better ($>0.85$).
3. **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures perceptual distance via VGG/AlexNet features. Lower is better ($<0.15$).
4. **NIQE (Natural Image Quality Evaluator)**: No-reference naturalness score. Lower is better ($<4.5$).
5. **ArcFace Cosine Identity**: Facial identity preservation score ($0..1$). Higher is better ($>0.70$).
