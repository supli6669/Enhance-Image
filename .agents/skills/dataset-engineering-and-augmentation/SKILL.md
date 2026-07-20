---
name: dataset-engineering-and-augmentation
description: Guidelines for face dataset preparation, 5-point landmark cropping and alignment at scale, LMDB binary storage conversion, dataset splitting, and data augmentation.
---

# Dataset Engineering & Augmentation Skill

Use this skill when preparing, filtering, cropping, aligning, or building binary storage datasets for machine learning models.

## Dataset Engineering Guidelines

1. **Facial Alignment at Scale**:
   - Use `facexlib` or `RetinaFace` to detect 5 facial landmarks across raw image folders.
   - Crop and align faces to exact $512 \times 512$ square dimensions (`crop_align_face.py`).
   - Filter out small face crops (< 100px) or low-confidence detections (< 0.6) to prevent dataset corruption.

2. **LMDB High-Throughput Binary Storage**:
   - Convert loose image files (PNG/JPG) to LMDB binary databases (`tools/build_lmdb.py`) when training on high-iteration deep learning models.
   - LMDB reduces disk I/O bottlenecks by enabling sequential memory-mapped binary reads instead of random file system lookups.

3. **Data Splitting & Domain Mixing**:
   - Split datasets into Train (90%), Validation (5%), and Test (5%) sets.
   - When fine-tuning on custom domains (e.g. game characters, stylized portraits), mix custom domain crops (30-50%) with general high-res face datasets (FFHQ 50-70%) to preserve generalizability.
