---
name: blind-degradation-pipeline
description: Synthesizing real-world image degradation pipelines including anisotropic Gaussian blur, motion kernels, Poisson/Gaussian noise, JPEG compression artifacts, and multi-stage downsampling.
---

# Blind Degradation Pipeline Modeling Skill

## Overview
Training robust face restoration models requires simulating realistic real-world corruptions. A multi-stage degradation pipeline combines blur, noise, downsampling, and JPEG compression in randomized orders.

## Degradation Order
1. **Blur**: Anisotropic Gaussian Blur ($\sigma \in [0.2, 3.0]$) + Motion blur kernel.
2. **Downsampling**: Random resize (area, bilinear, bicubic) down to scale factor $r \in [1, 8]$.
3. **Noise**: Additive Gaussian noise ($\sigma \in [0, 30]$) + Poisson noise.
4. **JPEG Compression**: Quality factor $Q \in [10, 70]$.
5. **Final Downscale**: Resize to network input shape ($512 	imes 512$).
