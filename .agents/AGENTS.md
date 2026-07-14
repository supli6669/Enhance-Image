# Workspace Rules: Custom AI Enhancer CPU Optimization Guidelines

All AI agents working on this codebase must adhere strictly to these rules:

1. **Read Handover Documentation**:
   - Always read the [handover.md](file:///d:/.gemini-scratch/custom-ai-enhancer/handover.md) file first to understand the historical context and architecture of the project.

2. **Strict CPU Performance Optimization Constraint**:
   - This project is deployed and tested on CPU-only environments (CUDA is unavailable).
   - Any pipeline updates or additions must prioritize CPU processing speed. Do **NOT** default to heavy deep-learning processes.

3. **No Redundant Deep-Learning Face Upscaling**:
   - The restored face output by CodeFormer is already high-quality and clean at $512 \times 512$ resolution.
   - Do **NOT** run Real-ESRGAN or any other neural network model on the restored face crops to upscale them by default. This takes over **60 seconds per face** on CPU.
   - Always use Lanczos interpolation (`cv2.INTER_LANCZOS4`) by default to resize and warp the face back. It takes only **0.016s** (a **3,800x speedup**).
   - Only execute neural-network face upscaling if the parameter `face_upsample=True` is explicitly passed (controlled by the user via the "Real-ESRGAN Face Upscale" toggle).

4. **Default to Faster Face Detectors on CPU**:
   - Always default to `retinaface_mobile0.25` or `YOLOv5n` in the user interface and processing configurations.
   - Do **NOT** set `retinaface_resnet50` as the default face detector as it takes **3.2s** on CPU compared to **0.1s** for the mobile variant.

5. **Optimize ONNX Runtime Session Options**:
   - When using ONNX Runtime sessions for inference on CPU, always construct and pass `SessionOptions` with `GraphOptimizationLevel.ORT_ENABLE_ALL` to maximize performance.
