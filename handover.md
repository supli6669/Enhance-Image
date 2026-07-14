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


