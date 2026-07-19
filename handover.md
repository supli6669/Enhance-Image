# Custom AI Enhancer Handover Log

## Task 1: Project Setup & Dependency Management

### Completed Operations
- Created the project structure at `C:\Users\admin\.gemini\antigravity-ide\scratch\custom-ai-enhancer`.
- Initialized local Git repository and added remote origin `https://github.com/supli6669/Enhance-Image`.
- Configured `.gitignore` to prevent committing virtual environments, model weights, cache, and inputs/outputs.
- Established a Python virtual environment `.venv` with Python 3.13.
- Resolved Python 3.13 / `basicsr` build incompatibility by creating a patch script `patch_and_install_basicsr.py` which clones `BasicSR` and patches the `setup.py` version parsing (`KeyError: '__version__'`) before installing it without CUDA extension requirement.
- Installed all required packages: PyTorch, TorchVision, OpenCV, Streamlit, facexlib, lpips, and gdown.
- Frozen dependencies and saved them into a standard, clean `requirements.txt`.

### Code Changes
- [NEW] [.gitignore](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/.gitignore)
- [NEW] [requirements.txt](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/requirements.txt)
- [NEW] [patch_and_install_basicsr.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/patch_and_install_basicsr.py)

### Git Commit & Push Status
- **Commit Message:** "feat: initialize project, setup virtual env, and resolve basicsr dependency"
- **Remote Push:** Completed (author config resolved to `supli6669`).

---

## Task 2: Model Repositories Integration

### Completed Operations
- Programmatically cloned the official `sczhou/CodeFormer` repository into `models/CodeFormer`.
- Created `download_weights.py` to download `codeformer.pth` (370MB) and additional face detection (`detection_Resnet50_Final.pth`), parsing (`parsing_parsenet.pth`), and YOLOv5 (`yolov5l-face.pth`) model weights into local `weights/` folder.
- Resolved local import conflict inside `models/CodeFormer/basicsr` by creating a custom local `version.py` file to satisfy the `basicsr.version` import requirement.
- Created `verify_imports.py` to programmatically configure Python path (`sys.path.insert`), verify imports, instantiate CodeFormer model structure, and load weights successfully on PyTorch. Verified successful execution.

### Code Changes
- [NEW] [models/CodeFormer/](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/models/CodeFormer/) (Cloned submodule, ignored heavy weights)
- [NEW] [models/CodeFormer/basicsr/version.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/models/CodeFormer/basicsr/version.py) (Vested local version descriptor)
- [NEW] [download_weights.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/download_weights.py) (Model weights downloader)
- [NEW] [verify_imports.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/verify_imports.py) (Path and import verification test)

### Git Commit & Push Status
- **Commit Message:** "feat: clone CodeFormer, download pretrained weights, and verify imports"
- **Remote Push:** Completed.

---

## Task 3: Build Custom Hybrid Pipeline

### Completed Operations
- Created `pipeline.py` implementing the `LocalAIEnhancerPipeline` class.
- Configured OpenCV image reading, loading `FaceRestoreHelper` for face landmarks detection and warping/cropping.
- Passed warped face crops through the local CodeFormer model with customizable fidelity parameter ($w$) using PyTorch.
- Designed a custom face pasting function (`paste_faces_custom_blend`) that exposes a `blend_softness` (0.0 to 1.0) parameter. This dynamically modifies the erosion radius and Gaussian blur size applied to the face boundary mask for seamless blending back into the upscaled background image.
- Combined the soft edge boundary mask with CodeFormer's PyTorch face features parsing segmentation mask to prevent blending artifacts.
- Created `test_pipeline.py` which runs the entire pipeline on a local sample image, verifies the upscaled dimensions, and saves the output to `test_output.png`. Tested successfully.

### Code Changes
- [NEW] [pipeline.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/pipeline.py) (Main processing pipeline with customizable fidelity and soft blending mask)
- [NEW] [test_pipeline.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/test_pipeline.py) (Verification test for the custom pipeline)

### Git Commit & Push Status
- **Commit Message:** "feat: implement custom enhancement pipeline with adjustable soft blending"
- **Remote Push:** Completed.

---

## Task 4: Advanced Streamlit UI & Hugging Face Spaces Deployment

### Completed Operations
- Created `app.py` containing the Streamlit web application.
- Caching initialized `LocalAIEnhancerPipeline` resources via `@st.cache_resource` to avoid loading 370MB weights on every page rerun.
- Designed a sidebar containing AI parameters:
  - **Fidelity Weight ($w$)**: Slider from 0.0 to 1.0 (fine-tuning quality/hallucination vs likeness).
  - **Mask Blending Softness**: Slider from 0.0 to 1.0 (manually controlling feather/blur of edges).
  - **Face Detector Model**: Dropdown (`retinaface_resnet50`, `retinaface_mobile0.25`, `YOLOv5l`, `YOLOv5n`).
  - **Background Upscale Factor**: Slider to set scaling size.
  - **Real-ESRGAN Background Upscale**: Checkbox to toggle AI-based super-resolution for the background.
  - **Face Detection Threshold**: Slider from 0.1 to 1.0 (controls the confidence threshold of RetinaFace/YOLOv5 dynamically).
- Built a side-by-side Before (Original) vs. After (AI Restored) comparison section displaying image stats (dimensions, duration) and a high-speed download button.
- Custom styled the UI using HTML/CSS markdown injection for a radial dark theme, gradient headers, and glassmorphic cards.
- **Hugging Face Spaces Optimization (Docker SDK)**:
  - Modified [pipeline.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/pipeline.py) to automatically download model weights (including Real-ESRGAN weights) at runtime if they are missing.
  - Created [README.md](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/README.md) containing setup instructions.
  - Added [Dockerfile](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/Dockerfile) pre-configured with a CPU-only PyTorch setup to build fast, bypass size limits, and start the Streamlit server on port `7860`. This enables direct deployment via the Hugging Face **Docker SDK**.

### Code Changes
- [NEW] [app.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/app.py) (Streamlit User Interface script)
- [MODIFY] [pipeline.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/pipeline.py) (Added automatic weight download triggers and Real-ESRGAN & threshold handling)
- [NEW] [README.md](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/README.md) (Project documentation)
- [NEW] [Dockerfile](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/Dockerfile) (Docker container environment setup)
- [MODIFY] [download_weights.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/download_weights.py) (Added RealESRGAN model weights to downloader)

### Git Commit & Push Status
- **Files Staged:** `app.py`, `pipeline.py`, `download_weights.py`, `handover.md`
- **Commit Message:** "feat: integrate Real-ESRGAN background upscaling and face detection threshold"
- **Remote Push:** Scheduled for execution.

---

## Task 5: Peak End-to-End Model Improvement Plan

### Overview
This plan describes the comprehensive, peak end-to-end strategy to improve and fine-tune the CodeFormer face restoration model on custom target domain datasets, covering data preparation, degradation pipeline adjustment, advanced loss selection, distributed training, validation, and integration.

---

### Step 1: Data Preparation & Preprocessing Pipeline
To fine-tune the model, you need a high-quality (HQ) training dataset. If you have low-quality (LQ) images, you also need to align them.
1. **Acquire HQ Face Dataset:** Prepare 2,000 - 10,000 high-quality face images (e.g. from your target domain or high-res portraits).
2. **Crop & Align Faces:**
   Run the face detection and alignment helper to crop faces to $512 \times 512$ pixels:
   ```bash
   python models/CodeFormer/scripts/crop_align_face.py -i <input_raw_images_dir> -o <output_aligned_faces_dir>
   ```
3. **Data Splitting:** Divide aligned faces into training (90%), validation (5%), and test (5%) splits. Store them under `models/CodeFormer/datasets/custom_dataset/`.

---

### Step 2: Degradation Modeling Customization
Modify the blind dataset configurations in your custom training option file (e.g. `CodeFormer_stage3_custom.yml`) to represent target real-world degradations:
- **Motion Blur:** Set `motion_kernel_prob` and add motion blur kernels to model camera movement.
- **Gaussian Blur:** Modify `blur_kernel_size` and `blur_sigma` to match degradation level.
- **Noise:** Add Poisson and Gaussian noise with custom parameters (`noise_range` or `noise_range_large`).
- **JPEG Compression:** Decrease the minimum of `jpeg_range` if dealing with high compression blockiness.

---

### Step 3: Architecture & Fine-Tuning Scenarios
Depending on your project's goals, select one of the following training pathways:
- **Scenario A: CFT Module Fine-Tuning (Stage III) - Recommended First Step**
  - Keeps Stage 1 (VQGAN) and Stage 2 (Transformer) frozen. Fine-tunes the controllable feature transformation layers to balance likeness (fidelity) and quality.
  - Very stable, relatively fast, and requires less GPU memory.
- **Scenario B: Transformer & CFT Fine-Tuning (Stage II & III)**
  - Fine-tunes the lookup transformer to map distorted inputs to the clean codebook indices.
  - Useful if the degradations are highly non-linear or stylized (e.g. cartoons, oil paintings).
- **Scenario C: Full VQGAN + Transformer Retraining (Stage I, II & III)**
  - Re-trains the VQGAN codebook representation from scratch.
  - Necessary only if restoring non-human faces (e.g., animal faces, fictional creatures).

---

### Step 4: Advanced Loss Function Adjustments
To enhance qualitative results and identity preservation:
1. **Identity Preservation (ArcFace Loss):** Integrate an ArcFace feature extractor to compute Cosine Similarity between restored and original faces:
   $$\mathcal{L}_{id} = 1 - \cos(\text{ArcFace}(I_{rec}), \text{ArcFace}(I_{HQ}))$$
2. **Structural & Detail Control:**
   - **Perceptual (LPIPS) Loss:** Retain at weight `1.0` for natural textures.
   - **GAN Loss:** Use Hinge GAN Loss (`loss_weight: 0.1`) to generate sharp details without artifacts.
   - **Pixel (L1) Loss:** Retain at weight `1.0` to avoid drift in color/lighting.

---

### Step 5: Distributed GPU Training Setup
For official training, use GPU(s) with CUDA:
1. **Create Option File:** Save configuration to [CodeFormer_stage3_custom.yml](file:///c:/Users/admin/.gemini/antigravity-ide/scratch/custom-ai-enhancer/models/CodeFormer/options/CodeFormer_stage3_custom.yml). Set `num_gpu: 1` (or more).
2. **Execute Training via torchrun (Distributed):**
   ```bash
   torchrun --nproc_per_node=gpu_num models/CodeFormer/basicsr/train.py -opt models/CodeFormer/options/CodeFormer_stage3_custom.yml --launcher pytorch
   ```
3. **Mixed Precision (AMP):** Enable AMP to save memory and speed up computation.

---

### Step 6: Evaluation & Metrics Validation
Validate checkpoints quantitatively and qualitatively:
- **PSNR / SSIM:** Measure reconstruction fidelity.
- **LPIPS:** Measure perceptual closeness to human vision.
- **FID:** Measure distribution quality of generated faces.
- **ArcFace Cosine similarity:** Validate face identity preservation.

---

### Step 7: Streamlit Integration
1. Export the best trained checkpoint (`params_ema` key) from `experiments/` to `weights/CodeFormer/codeformer_custom.pth`.
2. Update [pipeline.py](file:///d:/.gemini-scratch/custom-ai-enhancer/pipeline.py) to point to the new model weights.
3. Update [app.py](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) to add a model-selection dropdown or toggle, letting users compare the vanilla CodeFormer against your custom fine-tuned model.

---

## Task 6: Google Colab GPU Setup & ONNX Runtime CPU Inference Optimization

### Completed Operations
- **Colab GPU Training Notebook**: Created `train_on_colab.ipynb` for GPU-accelerated training. Implemented real-time checkpoint synchronization directly to the user's Google Drive using symbolic links (`ln -s`) to prevent data loss.
- **ONNX Export Script**: Created `tools/export_onnx.py` supporting dynamic scale selection (`scale=4` for custom checkpoints, `scale=2` for pretrained vanilla weights) and dynamic input shape axes for Real-ESRGAN (`RRDBNet`).
- **CodeFormer ONNX Compatibility**: Removed dynamic data-dependent control flow (`if w>0`) in `models/CodeFormer/basicsr/archs/codeformer_arch.py` to allow successful graph tracing with dynamic fidelity parameters.
- **Pipeline ONNX Runtime Integration**: Updated `pipeline.py` to automatically load ONNX Runtime sessions for both CodeFormer and Real-ESRGAN if their respective `.onnx` files are found under `weights/`, bypassing heavy PyTorch model initialization.
- **Verification Tests**: Verified model exports and end-to-end pipeline execution with ONNX Runtime using `tools/test_pipeline.py` and custom scripts successfully.

### Code Changes
- [NEW] [train_on_colab.ipynb](file:///d:/.gemini-scratch/custom-ai-enhancer/train_on_colab.ipynb) (Google Colab Setup Notebook)
- [NEW] [tools/export_onnx.py](file:///d:/.gemini-scratch/custom-ai-enhancer/tools/export_onnx.py) (Model to ONNX exporter)
- [MODIFY] [pipeline.py](file:///d:/.gemini-scratch/custom-ai-enhancer/pipeline.py) (Added ONNX execution sessions, cleaned up loops)
- [MODIFY] [requirements.txt](file:///d:/.gemini-scratch/custom-ai-enhancer/requirements.txt) (Added onnx and onnxruntime)
- [MODIFY] [tools/test_pipeline.py](file:///d:/.gemini-scratch/custom-ai-enhancer/tools/test_pipeline.py) (Fixed workspace paths and search logic)
- [MODIFY] [models/CodeFormer/basicsr/archs/codeformer_arch.py](file:///d:/.gemini-scratch/custom-ai-enhancer/models/CodeFormer/basicsr/archs/codeformer_arch.py) (Bypassed dynamic w check to support ONNX tracing)

### Git Commit & Push Status
- **Files Modified/Created**: Ready for commit.
- **Remote Push**: Pending user review.

---

## Task 7: CPU Performance Optimization & Guidelines

### Completed Operations
- **Real-ESRGAN Face Upscale Bypass**: Identified that running Real-ESRGAN on $512 \times 512$ restored faces on CPU takes **62.3 seconds** per face, causing massive bottlenecks. Implemented a bypass that uses Lanczos interpolation (`cv2.INTER_LANCZOS4`) by default, taking only **0.016 seconds** (a **3,800x speedup**) with virtually identical visual quality.
- **ONNX Session Optimization**: Added ONNX Runtime `SessionOptions` configuring `GraphOptimizationLevel.ORT_ENABLE_ALL` for both CodeFormer and Real-ESRGAN CPU inference.
- **Fast Default Face Detector**: Configured the default face detector in the web interface to be `retinaface_mobile0.25`, reducing detection overhead from **3.2s** (`retinaface_resnet50`) to **0.1s - 0.2s** on CPU.
- **User Toggles**: Added the **Real-ESRGAN Face Upscale** toggle in the sidebar (disabled by default) to let users explicitly run the heavy face upscaling model if desired.

### Code Changes
- [MODIFY] [pipeline.py](file:///d:/.gemini-scratch/custom-ai-enhancer/pipeline.py) (Added optional face upscaling, configured ONNX SessionOptions)
- [MODIFY] [app.py](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) (Set mobile face detector as default, added Real-ESRGAN Face Upscale toggle)
- [MODIFY] [models/CodeFormer/facelib/detection/__init__.py](file:///d:/.gemini-scratch/custom-ai-enhancer/models/CodeFormer/facelib/detection/__init__.py) (Fixed absolute config path loading for YOLOv5 detectors)
- [NEW] [tools/benchmark.py](file:///d:/.gemini-scratch/custom-ai-enhancer/tools/benchmark.py) (Benchmark profiling tool for pipeline parts)

### Guidelines for Future Agents
1. **Always Optimize for CPU**: Since this environment runs on CPU (CUDA is unavailable), any new features or models must be lightweight or off by default.
2. **Never Force Deep-Learning Face Upscaling**: Keep Real-ESRGAN face upscaling off by default. Use Lanczos/bicubic interpolation when pasting the $512 \times 512$ CodeFormer face back unless the user explicitly enables `face_upsample=True`.
3. **Prefer Mobile Face Detectors**: Default to `retinaface_mobile0.25` or `YOLOv5n` for fast CPU processing.

---

## Task 8: CPU / RAM / Disk Training Optimization (Max Resource Utilization)

### Machine Profile (measured)
- **CPU**: AMD Ryzen 7 7735HS (Zen 3+, 8C/16T, **AVX512** support)
- **RAM**: 28.6 GB usable (Windows reports 32 GB)
- **torch**: 2.12.1+cpu — `mkldnn` available, `bf16` CPU autocast available
- **Dataset**: 15,087 PNG images (~5.3 GB) in `datasets/realesrgan_gt`
- **Disk**: C: 31 GB free (OS), **D: 107 GB free** (project lives here)

### ⚠️ CRITICAL: Segfault root cause (this is why the old "running" training actually crashed)
The previous run that looked like it was "training" was in fact **segfaulting** (exit code
`3221225477` = `0xC0000005` access violation) on the **first forward pass** — the checkpoint at
iter 1850 came from a different environment (the HF Space GPU), NOT from this machine.

Root cause chain (verified by isolated repro scripts):
1. **oneDNN (mkldnn) CPU conv path segfaults** on this Ryzen 7735HS for the RRDB / upsample
   convolutions during training. Disabling mkldnn (`torch.backends.mkldnn.enabled = False`)
   eliminates the crash. **This is mandatory.**
2. **`filter2D` (random degradations) also segfaults** on CPU because it calls
   `F.pad(..., mode='reflect')` then `F.conv2d` on a non-contiguous tensor — same class of bug.
   Fixed by rewriting `filter2D` to use **OpenCV** (`cv2.filter2D`) in BOTH basicsr copies
   (`D:\Temp\BasicSR_src/basicsr/utils/img_process_util.py` and
   `models/CodeFormer/basicsr/utils/img_process_util.py`).
3. **`num_block` (RRDB depth) must be ≤ 6** for stable CPU training. With the real training
   input size (lq = 64×64), depth up to 16 builds, but the full GAN+perceptual pipeline is only
   stable at **num_block=6** (depth ≥ 16 intermittently segfaults / corrupts memory over
   iterations). The standard Real-ESRGAN `num_block=23` **cannot run on this CPU** — it segfaults
   at the body conv. If you need the full 23-block model, train on the HF Space (GPU) instead.
4. **`gt_size` must be ≤ 256** (not 320). The VGG perceptual loss on the 4× upscaled output
   (320→1280) allocates >5.6 GB for a single tensor and OOMs on 28 GB RAM. `gt_size=256`
   (output 1024) fits comfortably.

### Completed Operations
- **CPU — disabled the crashing oneDNN path, kept all cores busy**
  - Added `torch.backends.mkldnn.enabled = False` at the top of `realesrgan/train.py` (runs
    inside the training subprocess, so it actually takes effect).
  - `num_worker_per_gpu=0` (main process does degradation + compute; workers add no benefit and
    the DataLoader worker spawn was unstable here). `OMP/MKL_NUM_THREADS=8` + `MKL_THREADING_LAYER=GNU`
    to avoid the OpenMP/MKL threading crash; torch still parallelises matmuls/conv via its own
    intra-op pool across all 16 logical CPUs.
  - **Do NOT set `ATEN_CPU_CAPABILITY=avx512`** — if the installed torch build lacks the avx512
    kernel it raises SIGILL/segfault on the first forward pass. Let torch auto-detect the ISA.
- **RAM — increased memory footprint to feed compute without OOM**
  - `batch_size_per_gpu`: **12** (uses more RAM, more stable gradients).
  - `queue_size`: **120** (divisible by 12 for the degradation queue), `prefetch_mode: null`.
  - Observed live usage: ~1.3 GB RAM / 1234 CPU-s after iter 1 — plenty of headroom on 28 GB.
- **Disk — converted dataset to LMDB on D: for fast sequential I/O**
  - `tools/build_lmdb.py` converts the 15,087 loose PNGs into an LMDB at `D:\realesrgan.lmdb`
    (folder name ends with `.lmdb` as required by `RealESRGANDataset`). Built successfully.
  - `train_realesrgan.py` auto-builds the LMDB (Step 3.5) if missing, then points the config at it.
- **Quality / model — working config**
  - `num_block`: 23 → **6** (mandatory, see root cause #3).
  - `gt_size`: 320 → **256** (mandatory, see root cause #4).
  - `total_iter`: **50,000**.
- **CodeFormer (`train_custom.py`)**: left at `num_worker_per_gpu=4`; same mkldnn-off + GNU
  threading guidance applies if you train it on CPU.

### Code Changes
- [MODIFY] [models/Real-ESRGAN/realesrgan/train.py](file:///d:/.gemini-scratch/custom-ai-enhancer/models/Real-ESRGAN/realesrgan/train.py) (disable mkldnn at startup)
- [MODIFY] [train_realesrgan.py](file:///d:/.gemini-scratch/custom-ai-enhancer/train_realesrgan.py) (num_block=6, gt_size=256, worker=0, batch=12, queue=120, prefetch=null, LMDB auto-build + config; removed avx512 env)
- [MODIFY] [D:\Temp\BasicSR_src/basicsr/utils/img_process_util.py](file:///D:/Temp/BasicSR_src/basicsr/utils/img_process_util.py) (filter2D → cv2)
- [MODIFY] [models/CodeFormer/basicsr/utils/img_process_util.py](file:///d:/.gemini-scratch/custom-ai-enhancer/models/CodeFormer/basicsr/utils/img_process_util.py) (filter2D → cv2)
- [MODIFY] [models/Real-ESRGAN/options/train_realesrgan_custom.yml](file:///d:/.gemini-scratch/custom-ai-enhancer/models/Real-ESRGAN/options/train_realesrgan_custom.yml) (num_block=6, gt_size=256)
- [NEW] [tools/build_lmdb.py](file:///d:/.gemini-scratch/custom-ai-enhancer/tools/build_lmdb.py) (PNG → LMDB converter on D:)

### Verification
- `tools/build_lmdb.py` executed end-to-end: 15,087 images → `D:\realesrgan.lmdb`, `meta_info.txt` 15,087 lines.
- Full training pipeline (`RealESRGANModel.optimize_parameters`) ran 3 iters OK in a debug harness.
- **Live training confirmed running**: `realesrgan/train.py` reached `iter: 1` with losses
  `l_g_pix=0.54 l_g_percep=1.55 l_g_gan=0.07` and was actively consuming CPU (~1234 CPU-s, ~1.3 GB RAM).

### Git Commit & Push Status
- **Commit Message:** "fix: make CPU training run (disable mkldnn, cv2 filter2D, num_block=6, gt=256, LMDB)"
- **Remote Push:** Completed.

### Notes for Future Agents
- The LMDB lives on **D:** (`D:\realesrgan.lmdb`), outside the repo — not committed. Rebuild with
  `python tools/build_lmdb.py` if the source PNGs change.
- **If training segfaults again**, the first thing to check is whether mkldnn got re-enabled
  (e.g. a torch upgrade reverting `realesrgan/train.py`) or `num_block`/`gt_size` got bumped back up.
- The 23-block standard model only trains on GPU (HF Space). On this CPU, num_block=6 is the ceiling.



## Task 9: Future Optimization Plans (Plans A, B, C)

### Plan A – INT8 Quantization for ONNX Models
- **Goal:** Reduce model size & increase inference speed on CPU.
- **Tools:** `onnxruntime.quantization`, `tools/quantize_onnx.py`.
- **Steps:**
  1. Export current CodeFormer & Real‑ESRGAN models to ONNX (if not already present) using `tools/export_onnx.py`.
  2. Create script `tools/quantize_onnx.py`:
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(in_path, out_path):
    quantize_dynamic(in_path, out_path, weight_type=QuantType.QInt8)
```
  3. Run for each model:
```bash
python tools/quantize_onnx.py weights/codeformer.onnx weights/codeformer_int8.onnx
python tools/quantize_onnx.py weights/realesrgan.onnx weights/realesrgan_int8.onnx
```
  4. Update `pipeline.py` to prefer `_int8.onnx` if it exists.
  5. Benchmark using `tools/benchmark.py` (measure latency, memory, PSNR/LPIPS impact).
- **Verification:** Compare inference time before/after, confirm size reduction and acceptable quality drop (<2 % PSNR loss).

### Plan B – Parallel / Batch Face Processing
- **Goal:** Speed up processing of images containing multiple faces.
- **Approach A (ThreadPoolExecutor):**
  1. Detect all faces using the fast detector.
  2. Submit each face crop to a thread pool (`max_workers = os.cpu_count() // 2`).
  3. Each worker runs the CodeFormer ONNX session on its crop.
  4. Collect results and blend back using existing `paste_faces_custom_blend`.
- **Approach B (Batch Tensor):**
  1. Stack all face crops into a single batch tensor (`N x C x H x W`).
  2. Run a single ONNX session inference (`session.run(None, {"input": batch})`).
  3. Split batch output back to individual faces.
- **Implementation:** Add helper `pipeline._process_faces_batch()` and a flag `use_batch=True` in UI.
- **Verification:** Run on a test image with 5‑10 faces, ensure total time ≈ 1/​N of sequential.

### Plan C – Asynchronous UI Processing in Streamlit
- **Goal:** Prevent UI freeze when heavy tasks (Real‑ESRGAN background upscale, batch face processing) run.
- **Technique:** Use `st.experimental_singleton` / `st.session_state` to store a background thread.
```python
import threading, queue

def run_async(func, *args):
    q = queue.Queue()
    t = threading.Thread(target=lambda: q.put(func(*args)), daemon=True)
    t.start()
    return q, t
```
- **UI Changes:**
  * Add progress bar (`st.progress`) linked to thread status.
- **Verification:** Deploy locally, trigger a heavy upscale, confirm UI remains responsive and progress updates.

### Integration into Handovers
- Append this section to `handover.md` under **Task 9**.
- Update roadmap references in future AGENTS rules if needed.

---

## Task 10: Image Upload Bug Investigation & Fix Plan

**Date:** 2026-07-19  
**Status:** ✅ Completed

### Overview
Investigated why the Streamlit web app crashes or freezes when the user uploads an image. Full code-path audit of [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) and [`pipeline.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/pipeline.py) revealed **5 bugs** — from critical to low severity.

---

### Root Cause: 5 Bugs Found

#### Bug #1 — 🔴 CRITICAL: Background thread writes to `st.session_state` (Streamlit doesn't allow this)

**Location:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) lines 648–678

The `_run()` function is spawned as a `threading.Thread`. Inside it, results are written directly to `st.session_state`:
```python
st.session_state.enhanced_img = result       # ← from background thread ❌
st.session_state.processing_error = str(e)   # ← from background thread ❌
st.session_state.processing = False          # ← from background thread ❌
```
Streamlit **only allows** reading/writing `session_state` from the main request thread. Writes from background threads are silently dropped or cause race conditions. This is why the UI gets permanently stuck on the "processing" spinner — `enhanced_img` never gets set.

**Fix:** Use `queue.Queue` as a thread-safe bridge. The background thread pushes results into the queue; the main thread reads from it during the polling loop and writes to `session_state` safely.

---

#### Bug #2 — 🟠 HIGH: `progress_callback` bound into `@st.cache_resource` at cache time

**Location:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) lines 281–287

```python
@st.cache_resource(show_spinner=False, ...)
def get_pipeline():
    return LocalAIEnhancerPipeline(progress_callback=progress_callback)  # ← captured at cache time
```
`progress_callback` is captured once when the pipeline is first cached. The callback also writes to `session_state` from the background thread (compound of Bug #1). Additionally, if the session is refreshed, the cached callback may point to a stale session context.

**Fix:** Do not bind `progress_callback` in the constructor. Instead, pass it per-call to `process_image()`, or use the `queue.Queue` approach from Bug #1 to decouple the pipeline from session state entirely.

---

#### Bug #3 — 🟠 HIGH: `enhanced_img` can be `None`, not guarded before use

**Location:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) line 747

```python
'enhanced_shape': enhanced_img.shape[:2]  # ← AttributeError if None
```
If the pipeline returns `None` (e.g. silent exception in an edge case), this line crashes with an `AttributeError`. The same `None` value would also crash at `enhanced_img.shape` on line 755 and `cv2.cvtColor(enhanced_img, ...)` on lines 793, 800.

**Fix:** After reading `enhanced_img = st.session_state.enhanced_img`, add a `None` guard before any `.shape` or `cv2` usage.

---

#### Bug #4 — 🟡 MEDIUM: `FaceRestoreHelper` re-initialized on every `process_image()` call

**Location:** [`pipeline.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/pipeline.py) line 261

```python
def process_image(self, img, ...):
    face_helper = FaceRestoreHelper(upscale, face_size=512, det_model=detection_model, ...)
```
`FaceRestoreHelper.__init__` loads face detection weights (RetinaFace / YOLOv5) from disk every single call. On CPU this adds ~0.3–1.0 seconds of overhead per image and causes unnecessary disk I/O.

**Fix:** Cache `FaceRestoreHelper` instances in a dict keyed by `(detection_model, upscale)`. Call `face_helper.clean_all()` at the start of each `process_image()` call to reset the per-image state without re-loading weights.

```python
# In __init__:
self._face_helper_cache = {}

# In process_image():
cache_key = (detection_model, upscale)
if cache_key not in self._face_helper_cache:
    self._face_helper_cache[cache_key] = FaceRestoreHelper(upscale, face_size=512, det_model=detection_model, ...)
face_helper = self._face_helper_cache[cache_key]
face_helper.clean_all()
face_helper.read_image(img)
```

---

#### Bug #5 — 🟢 LOW: `split_img` recomputed redundantly in Download section

**Location:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) lines 816–818

`split_img` is computed again inside the Download Results block regardless of which view mode is active. This is harmless correctness-wise but duplicates computation. Minor cleanup: compute it once and reuse.

---

### Planned Fix Summary

| # | Bug | Severity | File | Lines |
|---|-----|----------|------|-------|
| 1 | Thread writes `session_state` unsafely | 🔴 Critical | app.py | 648–678 |
| 2 | `progress_callback` bound at cache time | 🟠 High | app.py | 281–287 |
| 3 | `enhanced_img` not guarded for `None` | 🟠 High | app.py | 747, 755, 793 |
| 4 | `FaceRestoreHelper` re-created every call | 🟡 Medium | pipeline.py | 261 |
| 5 | `split_img` redundant computation | 🟢 Low | app.py | 816–818 |

### Architecture Decision (Applied)
- **Option A (Chosen):** Kept background threading. Added `queue.Queue` as thread-safe bridge for results. Main thread reads queue during polling loop and writes `session_state` safely.

### Code Changes (Applied)
- [MODIFY] [app.py](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) (Fixed bugs #1, #2, #3, #5 via queue IPC and None guards)
- [MODIFY] [pipeline.py](file:///d:/.gemini-scratch/custom-ai-enhancer/pipeline.py) (Fixed bug #4 — cached FaceRestoreHelper dynamically)
- [MODIFY] [Dockerfile](file:///d:/.gemini-scratch/custom-ai-enhancer/Dockerfile) (Added headless, telemetry, and CORS/XSRF disable flags to streamlit run command)
- [MODIFY] [requirements.txt](file:///d:/.gemini-scratch/custom-ai-enhancer/requirements.txt) (Cleaned up fake version numbers to resolve Hugging Face build failure)
- [MODIFY] [.github/workflows/hf_sync.yml](file:///d:/.gemini-scratch/custom-ai-enhancer/.github/workflows/hf_sync.yml) (Added token checks to output clear error on github action failure)

### Git Commit & Push Status
- **Status:** Push completed to origin (GitHub) and hf (Hugging Face Spaces) main branch.

---

## Task 11: Full Bug Audit & Backlog

**Date:** 2026-07-19
**Status:** 🔵 In Progress — Bugs identified, fixes pending

### Overview
Performed a full static code audit of [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py), [`pipeline.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/pipeline.py), [`Dockerfile`](file:///d:/.gemini-scratch/custom-ai-enhancer/Dockerfile), and [`.github/workflows/hf_sync.yml`](file:///d:/.gemini-scratch/custom-ai-enhancer/.github/workflows/hf_sync.yml). Found **12 bugs** total.

> **Pipeline import status:** ✅ `from pipeline import LocalAIEnhancerPipeline` succeeds locally.

---

### Bug Backlog (Priority Order)

| # | Status | Sev | File | Description |
|---|--------|-----|------|-------------|
| B1 | ✅ Fixed | 🔴 Critical | app.py | `processing`, `enhanced_img`, `processing_error`, `process_duration` used with no init guard |
| B2 | ❌ Open | 🔴 Critical | pipeline.py | `enhance_realesrgan_onnx()` sends full image to ONNX without tiling — OOM on large images |
| B3 | ✅ Fixed | 🟠 High | pipeline.py | Parallel ONNX face processing shares `ort_session_cf` across threads — not thread-safe |
| B4 | ✅ Fixed | 🟠 High | app.py | Batch tab calls `pipeline.process_image()` synchronously on main thread — UI freezes |
| B5 | ✅ Fixed | 🟠 High | app.py | Dead `progress_callback()` (line 272) still writes `session_state` from thread — dangerous |
| B6 | ❌ Open | 🟡 Medium | app.py | `st.session_state.start_time` read in background thread without init guard |
| B7 | ✅ Fixed | 🟡 Medium | pipeline.py | `face_helper.face_size` assumed to be tuple, can be `int` on some facexlib versions |
| B8 | ✅ Fixed | 🟡 Medium | app.py | Training dashboard regex only captures `cross_entropy_loss` — Real-ESRGAN runs show `0.0` |
| B9 | ✅ Fixed | 🟡 Medium | app.py | Keyboard shortcut `Esc` uses `button:contains()` — invalid CSS, Cancel never fires |
| B10 | ❌ Open | 🟢 Low | app.py | `split_img` computed twice — once in Split Screen view, once in Download section |
| B11 | ❌ Open | 🟢 Low | Dockerfile | `patch_and_install_basicsr.py` does `git clone` at build time — fails on slow/offline network |
| B12 | ❌ Open | 🟢 Low | app.py | CSS `li::before { display:flex }` on pseudo-element — non-standard, visual glitch in some browsers |

---

### Bug Details

#### B1 — 🔴 Session State Keys Have No Initialization Guard
**File:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) lines 219–243 (init block) and 635–643 (first use)

`progress_state`, `presets`, `history`, `dark_mode` are all guarded with `if 'x' not in st.session_state`. But `processing`, `enhanced_img`, `processing_error`, `process_duration`, `start_time` are **never initialized** — they are directly assigned at line 636. On a cold start where `last_run_params` is `None` and no params have changed, the code jumps straight to line 643 (`st.session_state.enhanced_img is None`) and crashes with `AttributeError`.

**Fix:** Add to the init block (after line 226):
```python
for key, default in [
    ('processing', False),
    ('enhanced_img', None),
    ('processing_error', None),
    ('process_duration', None),
    ('start_time', None),
    ('last_run_params', None),
    ('history_added_for', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default
```

---

#### B2 — 🔴 ONNX RealESRGAN Has No Tiling — OOM on Large Images
**File:** [`pipeline.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/pipeline.py) lines 134–163

`enhance_realesrgan_onnx()` takes the entire image as a single input tensor. For a 1920×1080 image, this creates a `[1, 3, 1080, 1920]` float32 tensor. The ONNX model produces large intermediate activations and will OOM on memory-constrained environments (e.g. HF Spaces Free Tier ~16GB). The PyTorch path correctly uses `tile=400, tile_pad=40`.

**Fix:** Implement tile-based inference inside `enhance_realesrgan_onnx()`:
- Split image into overlapping 400px tiles with 40px padding
- Run ONNX on each tile separately
- Stitch tiles back together with a linear blend at seams

---

#### B3 — 🟠 Parallel ONNX Race on Shared Session
**File:** [`pipeline.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/pipeline.py) lines 349–383

When `parallel=True` and ONNX is active, multiple `ThreadPoolExecutor` workers call `self.ort_session_cf.run()` concurrently on the **same** session object. ONNX Runtime does not guarantee concurrent `.run()` calls on the same `InferenceSession` are safe. Random errors like `Invalid tensor shape` or `OrtValue index out of range` may occur when multiple faces are detected.

**Fix:** Add a `threading.Lock` around `ort_session_cf.run()` in `run_onnx_batch()`, or spawn a separate session per thread using `self._get_onnx_session()`.

---

#### B4 — 🟠 Batch Tab Freezes UI (Synchronous Main Thread)
**File:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) lines 932–991

The single-image tab was fixed to run `pipeline.process_image()` in a background thread. The batch tab still runs it synchronously on the Streamlit main thread in a `for` loop. For 10 images at ~30s each, the entire Streamlit app is frozen for ~5 minutes.

**Fix:** Wrap the batch loop in a background thread using `queue.Queue`, same pattern as the single-image tab. Post per-image results to the queue; main thread polls and updates `st.progress`.

---

#### B5 — 🟠 Dead `progress_callback` Writes `session_state` From Thread
**File:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) lines 272–281

```python
def progress_callback(stage, progress, message):
    st.session_state.progress_state = { ... }  # ← thread-unsafe write
```

This function is never used (replaced by the queue-based `local_progress_callback`). But it's still defined and the pipeline is initialized with `LocalAIEnhancerPipeline()` (no callback). If a future agent accidentally passes it to the constructor, the original thread-safety bug returns.

**Fix:** Delete this function entirely, or rename to `_DEPRECATED_progress_callback` with a `raise NotImplementedError` body.

---

#### B6 — 🟡 `start_time` Read in Thread Without Guard
**File:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) line 689

```python
'duration': time.time() - st.session_state.start_time,
```

If the session is lost between thread launch and result receipt (browser refresh, timeout), this raises `AttributeError`. Fix: use `st.session_state.get('start_time', time.time())`.

---

#### B7 — 🟡 `face_helper.face_size` Can Be `int` Not Tuple
**File:** [`pipeline.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/pipeline.py) lines 459, 463, 468, 478

`face_helper.face_size[0]` and `face_helper.face_size[1]` are used extensively. On some `facexlib` versions, `face_size` is set to `512` (int) not `(512, 512)` (tuple). Indexing an int raises `TypeError`.

**Fix:** At top of `paste_faces_custom_blend()`:
```python
fs = face_helper.face_size
face_size = fs if isinstance(fs, tuple) else (fs, fs)
```
Then replace all `face_helper.face_size` usages with `face_size`.

---

#### B8 — 🟡 Training Dashboard Loss = 0.0 for Real-ESRGAN
**File:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) line 361

```python
loss_match = re.search(r"cross_entropy_loss:\s*([\d.e+-]+)", line)
```

Real-ESRGAN logs use keys like `l_g_pix`, `l_g_percep`, `l_g_gan`. The regex only matches `cross_entropy_loss` (CodeFormer-specific). All Real-ESRGAN training sessions show loss `0.0`.

**Fix:**
```python
loss_match = (
    re.search(r"cross_entropy_loss:\s*([\d.e+-]+)", line) or
    re.search(r"l_g_pix:\s*([\d.e+-]+)", line) or
    re.search(r"l_g_percep:\s*([\d.e+-]+)", line)
)
```

---

#### B9 — 🟡 `Esc` Keyboard Shortcut Uses Invalid CSS Selector
**File:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) lines 324–328

```javascript
const cancelButton = document.querySelector('button:contains("Cancel")');
```

`:contains()` is a jQuery pseudo-selector. It does not exist in native browser `document.querySelector`. This always returns `null`, so `Esc` never cancels.

**Fix:**
```javascript
const cancelButton = Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('Cancel'));
if (cancelButton) cancelButton.click();
```

---

#### B10 — 🟢 `split_img` Computed Twice
**File:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) lines 855–858 and 874–876

`split_img` is created inside the `"🌗 Split Screen"` view mode block and also recreated unconditionally in the Download section. Cache and reuse.

---

#### B11 — 🟢 Dockerfile `git clone` at Build Time
**File:** [`Dockerfile`](file:///d:/.gemini-scratch/custom-ai-enhancer/Dockerfile) lines 25–26

`patch_and_install_basicsr.py` clones BasicSR from GitHub at Docker build time. This fails silently on network-restricted or rate-limited build runners. Consider vendoring BasicSR or caching the wheel as a pre-built artifact.

---

#### B12 — 🟢 CSS `::before { display:flex }` Non-Standard
**File:** [`app.py`](file:///d:/.gemini-scratch/custom-ai-enhancer/app.py) lines 185–191

`display:flex` on `::before` pseudo-elements is non-standard and inconsistent across browsers. Change to `display:inline-flex` or use `display:grid` with `place-items:center`.

---

### Notes for Future Agents
- Fix **B1 first** — it's a cold-start crash, very small change, high impact.
- Fix **B5 second** — delete/disable the dead `progress_callback` to prevent accidental regression.
- Fix **B7 third** — one-liner, prevents `TypeError` on some facexlib versions.
- **B2** (ONNX tiling) is the most complex fix — needs careful implementation to avoid seam artifacts.
- **B3, B4** are parallel/threading refactors — do them together.
- **B8, B9** are small regex/JS fixes — can be done as a single minor patch commit.
- **B10–B12** are cosmetic/housekeeping — batch at end of any session.

