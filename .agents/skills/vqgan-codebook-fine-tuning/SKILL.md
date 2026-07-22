---
name: vqgan-codebook-fine-tuning
description: Fine-tuning VQGAN discrete codebooks, discrete code dictionary index lookup, stage I VQ autoencoders, and codebook matching for photo & illustration styles.
---

# VQGAN Codebook Fine-Tuning & Dictionary Matching Skill

## Overview
CodeFormer relies on a VQGAN codebook (Stage I) that encapsulates high-quality clean face priors. When restoring domain-specific images (game characters, vintage portraits), fine-tuning codebook vectors or adjusting codebook lookup weights improves naturalness and reduces vector quantization quantization loss.

## Key Components
1. **Codebook Size**: $N=1024$ codebook vectors of dimension $d=512$.
2. **Quantization Loss**: Commitment loss + Codebook update loss.
3. **Cross-Entropy Code Prediction Transformer**: Stage II maps low-quality features to discrete codebook indices.
