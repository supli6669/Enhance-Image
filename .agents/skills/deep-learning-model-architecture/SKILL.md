---
name: deep-learning-model-architecture
description: Expert guidelines for deep learning architectures, VQGAN codebook vector quantization, Swin Transformer / CFT feature transformation, Real-ESRGAN RRDB networks, loss weighting, gradient accumulation, TorchScript/ONNX export, and dynamic shape tracing.
---

# Deep Learning Architecture & Fine-Tuning Skill

Use this skill when designing neural network architectures, configuring loss functions, exporting ONNX/TorchScript models, or optimizing training loops.

## Core Deep Learning Principles

1. **CodeFormer Architecture & Controllable Feature Transformation (CFT)**:
   - CodeFormer combines a discrete Codebook VQGAN (Stage I), a Transformer sequence predictor (Stage II), and Controllable Feature Transformation (CFT) modules (Stage III).
   - **Fidelity Parameter ($w \in [0, 1]$)**: Controls the feature trade-off in CFT modules:
     - $w = 0.0$: Maximum perceptual quality & hallucination of missing details.
     - $w = 1.0$: Maximum fidelity & identity preservation to degraded input.
   - When exporting to ONNX Runtime, remove dynamic data-dependent control flow statements (`if w > 0`) inside the forward pass to enable clean symbolic tracing.

2. **Real-ESRGAN RRDB Net Super-Resolution**:
   - Residual-in-Residual Dense Blocks (RRDB) extract multi-scale deep features.
   - Set scale factors (`scale=2` or `scale=4`) explicitly in model constructors.
   - For ONNX export, specify dynamic axes for input spatial dimensions `(height, width)` to support arbitrary image sizes:
     ```python
     torch.onnx.export(
         model, dummy_input, onnx_path,
         input_names=['input'], output_names=['output'],
         dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'out_height', 3: 'out_width'}}
     )
     ```

3. **Loss Function Engineering & Weighting**:
   - **Pixel Loss ($\mathcal{L}_1$)**: Weight `1.0` — maintains global color and luminance alignment.
   - **Perceptual Loss ($\mathcal{L}_{percep}$)**: Weight `1.0` — extracts VGG-19 feature maps for natural textures.
   - **GAN Adversarial Loss ($\mathcal{L}_{gan}$)**: Weight `0.1` — Hinge loss discriminator for realistic high-frequency detail generation.
   - **Identity Loss ($\mathcal{L}_{id}$)**: Weight `0.5` — ArcFace cosine similarity to prevent facial identity drift during fine-tuning.

4. **Training Loop Stability & Optimization**:
   - Use Exponential Moving Average (`EMA`) for generator weights to produce smooth, non-flickering predictions.
   - Enable PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast`) during GPU training to halve memory footprint and accelerate compute.
