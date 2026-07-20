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

6. **Sync Obsidian Vault After Every Session**:
   - The project utilizes an Obsidian knowledge base vault at `D:\AgentBrain\` containing Rules and Skills.
   - At the end of any session where you modify project rules or instructions, run the sync script to update the vault:
     ```powershell
     powershell -ExecutionPolicy Bypass -File "D:\AgentBrain\sync.ps1"
     ```
   - This will copy the latest `AGENTS.md` to the vault's Rules directory and update the vault timestamp.
   - Vault location: `D:\AgentBrain\`

7. **Mandatory Sequential Model Improvement Roadmap**:
   - All model-related work MUST follow the sequential phase order defined in the task list artifact:
     `C:\Users\admin\.gemini\antigravity-ide\brain\0bf6bec8-6164-477e-a32d-6f0b9ef577c6\task.md`
   - **Before starting any model improvement task**, read `task.md` and identify the first incomplete task (`[ ]`).
   - Do **NOT** skip phases or implement out-of-order. Each phase builds on the previous:
     - **Phase 1** — Config fixes (yml) — COMPLETED ✅
     - **Phase 2** — Resume & complete training run (iter 2k → 20k)
     - **Phase 3** — ArcFace identity loss integration
     - **Phase 4** — Dataset expansion verification & game character mixing
     - **Phase 5** — Static INT8 ONNX quantization with calibration
     - **Phase 6** — Stage II transformer fine-tune (GPU only, HF Space/Colab)
     - **Phase 7** — A/B test UI & model selection dropdown
   - After completing a task, mark it `[x]` in `task.md` before proceeding to the next.
   - Reference the full improvement rationale in:
     `C:\Users\admin\.gemini\antigravity-ide\brain\0bf6bec8-6164-477e-a32d-6f0b9ef577c6\model_improvement_proposals.md`

8. **Always Update task.md During Model Work**:
   - Mark tasks `[/]` (in progress) when starting them.
   - Mark tasks `[x]` (done) when verified complete.
   - Add notes under each task if you discover important findings (e.g., actual iteration count, loss values, timing).
   - This ensures seamless handover between sessions and agents.
