---
name: wink-portrait-enhancement-quality
description: Comprehensive procedural guide for Wink-level portrait enhancement, parsing-guided face detail enhancement, frequency separation skin grain retention, eye/lip sparkle boosting, LAB CLAHE tone balance, and high-speed OpenCV/NumPy CPU vectorization.
---

# Wink-Level Portrait Enhancement & Quality Guidelines

This skill provides architectural guidelines, formulas, and operational procedures for achieving Wink/Meitu-level visual quality in AI face restoration pipelines.

---

## 🎨 Core Architectural Components

### 1. Frequency Separation (Real Skin Grain Retention)
- **Problem:** Neural network face restoration models (CodeFormer, GFP-GAN) tend to generate plastic, overly smooth, or "soapy" skin textures at $512 \times 512$.
- **Solution:** Extract high-frequency noise and texture details from the original un-restored cropped face using Gaussian blur subtraction, then blend them back onto the restored face crop.
- **Formula:**
  $$\text{Blur} = \text{GaussianBlur}(I_{\text{orig}}, k=(5,5), \sigma=0)$$
  $$\text{HighFreq} = I_{\text{orig}} - \text{Blur}$$
  $$I_{\text{final}} = I_{\text{restored}} + \alpha \cdot \text{HighFreq} \odot M_{\text{skin}}$$
- **Default Parameter:** $\alpha = 0.15$ (Skin Grain Retention).

---

### 2. Parsing-Guided Feature Enhancement (Eyes & Lips)
- **Segmentation Model:** `facexlib` `parsenet` ($512 \times 512$ parsing mask).
- **Mask Indices:**
  - `1`: Skin
  - `4`: Left Eye
  - `5`: Right Eye
  - `6`: Glasses
  - `11`: Upper Lip
  - `12`: Lower Lip
  - `13`: Inner Mouth
- **Eye Sparkle (Catchlight Boost):**
  - Dilate eye mask using $3 \times 3$ ellipse kernel.
  - Apply CLAHE (`clipLimit=2.0`, `tileGridSize=(4,4)`) on the $L$ channel of LAB color space inside the eye mask.
  - Apply localized Unsharp Masking ($\text{Sharp} = 1.3 \cdot I - 0.3 \cdot \text{Blur}$) with Gaussian mask feathering.
- **Lip Polish:**
  - Increase HSV saturation channel ($S \leftarrow \min(S \cdot 1.1, 255)$) inside lip mask (`11, 12, 13`).

---

### 3. LAB CLAHE Lighting & Tone Balance
- **Color Space:** Convert face crop to LAB ($L$: Lightness, $A$: Green-Red, $B$: Blue-Yellow).
- **Luminance Equalization:** Apply CLAHE (`clipLimit=1.5`, `tileGridSize=(8,8)`) on $L$ channel.
- **Soft Blend:** $\text{L}_{\text{final}} = 0.6 \cdot L + 0.4 \cdot L_{\text{CLAHE}}$ to avoid over-exposure or blown-out skin highlights.

---

## ⚡ Performance Constraints on CPU

1. **No Neural Nets in Post-Processing:** All texture injection, eye sparkle, and tone balancing MUST use vectorized NumPy & OpenCV C++ routines (`WinkQualityEnhancer`).
2. **Latency Guarantee:** Total post-processing overhead must be $< 0.05\text{s}$ per face crop on 8-core CPU.
3. **Safe Parsing Fallback:** If facial parsing fails or mask is unavailable, degrade gracefully to global LAB CLAHE + unsharp masking without crashing the execution loop.
