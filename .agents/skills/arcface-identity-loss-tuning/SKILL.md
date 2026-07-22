---
name: arcface-identity-loss-tuning
description: Integrating ArcFace/CosFace identity feature extractors into PyTorch GAN loss pipelines to preserve facial identity during fine-tuning.
---

# ArcFace Identity Loss Integration & Fine-Tuning Skill

## Overview
Standard pixel (L1) and perceptual (LPIPS) losses focus on surface appearance, which can drift human identity features during high-fidelity face restoration. ArcFace extracts deep 512-dimensional face embedding vectors. Computing cosine distance between restored and target embeddings guarantees identity preservation.

## Mathematical Formulation
$$\mathcal{L}_{id} = 1 - rac{	ext{ArcFace}(I_{rec}) \cdot 	ext{ArcFace}(I_{gt})}{\|	ext{ArcFace}(I_{rec})\| \|	ext{ArcFace}(I_{gt})\|}$$

## Code Snippet
```python
import torch
import torch.nn.functional as F

class ArcFaceLoss(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad = False
            
    def forward(self, pred, target):
        emb_pred = F.normalize(self.net(pred), p=2, dim=1)
        emb_gt = F.normalize(self.net(target), p=2, dim=1)
        return 1.0 - torch.sum(emb_pred * emb_gt, dim=1).mean()
```
