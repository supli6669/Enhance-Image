---
name: fast-cpu-vectorization-numpy
description: Optimizing Python image pipelines with C-vectorized NumPy operations, OpenCV inplace buffer operations, and OpenMP multi-threading controls for <20ms CPU latency.
---

# Fast CPU Vectorization & NumPy Optimization Skill

## Overview
Running image post-processing in Python loops creates severe latency bottlenecks. Vectorizing image math with NumPy broadcasting, OpenCV inplace buffers (`dst=out`), and contiguous memory layouts ensures lightning-fast execution (<20ms per face).

## Golden Rules
1. **Never Python Loops over Pixels**: Always use vectorized slice operations (`img[:, :, 0]`).
2. **Inplace OpenCV Buffers**: Pass output buffer directly `cv2.addWeighted(img1, 0.5, img2, 0.5, 0, dst=buffer)`.
3. **Memory Contiguity**: Call `np.ascontiguousarray()` before passing arrays to C/C++ extensions.
