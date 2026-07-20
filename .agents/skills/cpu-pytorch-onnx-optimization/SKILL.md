---
name: cpu-pytorch-onnx-optimization
description: Specialized rules and procedures for optimizing PyTorch and ONNX Runtime performance on CPU-only environments, preventing oneDNN/mkldnn segfaults, performing shape inference, tile-based super-resolution, and fast Lanczos face warp-back.
---

# CPU PyTorch & ONNX Runtime Optimization Skill

Use this skill whenever working on PyTorch or ONNX inference/training on CPU hardware without CUDA acceleration.

## Key CPU Performance Constraints & Rules

1. **Prevent oneDNN / mkldnn Segfaults on AMD Ryzen / CPU**:
   - In PyTorch training/inference scripts on CPU, set:
     ```python
     torch.backends.mkldnn.enabled = False
     ```
   - Do NOT set `ATEN_CPU_CAPABILITY=avx512` if the PyTorch build lacks the avx512 kernel. Let PyTorch auto-detect hardware capability.

2. **OpenCV Filter2D Replacement for CPU Stability**:
   - `F.pad(..., mode='reflect')` + `F.conv2d` on non-contiguous tensors can cause segfaults in `basicsr` on CPU.
   - Use `cv2.filter2D` instead for image degradation filtering.

3. **ONNX Provider Initialization**:
   - Always avoid requesting `OpenVINOExecutionProvider` on Windows unless `openvino.dll` is verified present.
   - Prefer:
     ```python
     providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
     ```
   - Always pass `SessionOptions` with `GraphOptimizationLevel.ORT_ENABLE_ALL`:
     ```python
     opts = ort.SessionOptions()
     opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
     session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
     ```

4. **Multi-Candidate Fail-Safe ONNX Loading**:
   - When loading quantized or base ONNX models, always iterate over candidates in a `try-except` block to prevent startup crashes if an ONNX model has shape inference or quantization errors:
     ```python
     for candidate in candidates:
         if os.path.exists(candidate):
             try:
                 session = ort.InferenceSession(candidate, sess_options=opts, providers=['CPUExecutionProvider'])
                 break
             except Exception as e:
                 print(f"Warning: Failed to load {candidate}: {e}")
     ```

5. **Fast Lanczos Face Warp-Back (3,800x Speedup)**:
   - Restored CodeFormer faces ($512 \times 512$) are already clean and crisp.
   - Do NOT run heavy neural upscaling (Real-ESRGAN) on individual face crops by default on CPU.
   - Use `cv2.INTER_LANCZOS4` for warping faces back into the background image.

6. **Tile-Based ONNX Super-Resolution**:
   - When running Real-ESRGAN on large images, process in overlapping tiles (e.g. 400px tile with 40px padding) to prevent RAM OOM.
