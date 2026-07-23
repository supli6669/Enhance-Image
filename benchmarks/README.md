# Face Restoration Benchmark

This directory is the fixed evaluation set for model-quality decisions. Images
in `inputs/` and `references/` are intentionally ignored by Git because they
may contain private portraits. Commit only `manifest.csv` metadata if it does
not expose sensitive information.

## Setup

1. Copy `manifest.example.csv` to `manifest.csv`.
2. Add 300-500 held-out portrait examples. Each `input_path` is the degraded
   input. Where available, `reference_path` is its aligned high-quality target.
3. Keep the same manifest for the baseline and every candidate checkpoint.
4. Run the validation gate before an evaluation or training decision:

   ```powershell
   python tools/evaluate_restoration.py --manifest benchmarks/manifest.csv --dry-run
   ```

The benchmark must never overlap the training dataset. Use `category` to make
sure blur, compression, low-light, old-photo and severe-damage cases are all
represented.

## Acceptance gate

A candidate may advance only if it has no regression in the identity review,
does not regress median PSNR/SSIM where reference images exist, and is preferred
by the side-by-side human review for artifacts around eyes, skin, teeth and hair.
