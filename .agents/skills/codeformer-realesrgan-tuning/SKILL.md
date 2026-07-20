---
name: codeformer-realesrgan-tuning
description: Procedural guide for training, fine-tuning, loss configuration (ArcFace identity loss, perceptual loss), dataset degradation pipeline tuning, and checkpoint export for CodeFormer face restoration and Real-ESRGAN super-resolution.
---

# CodeFormer & Real-ESRGAN Model Fine-Tuning Skill

Use this skill whenever configuring options files (`.yml`), dataset degradation pipelines, loss functions, or training scripts for CodeFormer or Real-ESRGAN.

## Training Configuration & Options Guidelines

1. **Learning Rate & Scheduler Alignment**:
   - Always match `scheduler.periods` to `total_iter`. For example, if `total_iter: 20000`, set `periods: [20000]` to ensure proper learning rate annealing.

2. **Realistic Degradation Pipeline**:
   - For real-world face restoration, configure degradations in `CodeFormer_stage3_custom.yml`:
     - `blur_kernel_size: 21`
     - `jpeg_range: [10, 70]` & `jpeg_range_large: [5, 50]`
     - `motion_kernel_prob: 0.15`
     - `noise_range: [0.0, 30.0]` & `downsample_range: [1.0, 20.0]`

3. **CPU Model Depth Constraints**:
   - Standard Real-ESRGAN uses `num_block: 23`. On CPU environments, set `num_block: 6` and `gt_size: 256` to prevent PyTorch C++ memory allocation crashes and OOM. Full 23-block training requires GPU acceleration (Google Colab / HF Space GPU).

4. **ArcFace Identity Preservation Loss (Phase 3)**:
   - Compute cosine similarity between ArcFace feature vectors of restored face $I_{rec}$ and ground-truth face $I_{HQ}$:
     $$\mathcal{L}_{id} = 1 - \cos(\text{ArcFace}(I_{rec}), \text{ArcFace}(I_{HQ}))$$
   - Set weight `identity_loss_weight: 0.5` in options file.

5. **Checkpoint Export & Deployment**:
   - Export checkpoint weights from `experiments/.../models/net_g_*.pth` using `params_ema` key.
   - Place exported weight in `weights/CodeFormer/codeformer_finetuned.pth`.
   - Update `pipeline.py` to support dynamic checkpoint selection.
