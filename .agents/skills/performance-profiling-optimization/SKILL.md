---
name: performance-profiling-optimization
description: Use when profiling execution bottlenecks, benchmarking latency/throughput, optimizing memory footprints, or vectorizing slow Python code.
---

# Performance Profiling & Optimization Skill

Use this skill when tasked with profiling, benchmarking, or accelerating execution speed and memory efficiency.

## Optimization Methodology

1. **Establish Baseline Benchmark**:
   - Measure latency (ms/iter), throughput (images/sec), and peak RAM/VRAM before making any changes.
   - Use standard benchmark scripts (`tools/benchmark.py` or `time.perf_counter()`).

2. **Identify Bottlenecks**:
   - Profile code to identify whether bottleneck is I/O-bound, CPU compute-bound, memory allocation-bound, or thread lock contention.
   - Focus optimization efforts exclusively on the top 1-2 bottleneck components (Amdahl's Law).

3. **Apply Targeted Vectorization & Caching**:
   - Replace Python loops with vectorized NumPy/OpenCV/PyTorch operations.
   - Reuse model sessions and persistent objects instead of re-instantiating inside function calls.
   - Apply ONNX graph optimizations (`ORT_ENABLE_ALL`) and INT8 quantization where applicable.

4. **Verify Speedup & Quality Retention**:
   - Re-run benchmark to measure exact speedup factor.
   - Verify output quality (PSNR/SSIM/LPIPS) to ensure optimization did not introduce regression.
