# Sharpening update — 2026-09-13

The user clarified that visibly sharpening the image is the primary requirement.
The previous default only applied mild clarity after resizing and often looked
almost unchanged. This update changes that processing path, rather than only
adding controls or deferring model loading.

## Behavior

- Pure Quality uses Adaptive Sharpness at 0.50; Natural Likeness keeps 0.35.
- On the non-reconstruction Lanczos path, extract fine and medium luminance
  detail in source coordinates. Estimate noise from diagonal high-pass responses,
  apply coring and reduce gain with increasing estimated noise.
- Resize the shared luminance increment separately from the image. Constrain it
  to local extrema before and after resizing, then to available channel headroom.
  Each output channel receives the same integer increment, preserving channel
  differences relative to the Lanczos baseline.
- Do not stack the old clarity filter on top. When dehaze/deblur or neural SR is
  explicitly selected, retain the existing output-stage clarity path.
- Strength zero disables the change. Flat colors, sharp step edges, very small
  inputs and invalid controls have explicit regression coverage.

CodeFormer, its fidelity/source blend and neural face upscaling defaults are
unchanged. This is stronger enhancement of existing edges, not full restoration
of detail absent from severely blurred inputs.

## Evidence

Initial development used six references with five conditions. A fixed check used
12 other validation references, excluding all references from the earlier
three/six-image diagnostics, with five conditions each: 60 cases. They remain
experimental validation images, not identity-reviewed production holdout.

The final filter fixed a one-intensity-level edge leak found by a synthetic test.
The same frozen check was repeated to verify that correction, without changing
strength or retuning from its results. Final evidence is in
`artifacts/enhance_quality_check/adaptive_sharpen_final/report.json`.

Median paired differences against previous Pure (clarity 0.35):

| Condition | PSNR delta, dB | SSIM delta | LPIPS relative delta | ArcFace delta |
| --- | ---: | ---: | ---: | ---: |
| Soft blur | +0.0396 | +0.00065 | -0.24% | +0.00005 |
| Severe blur/downsampling | +0.2469 | +0.00225 | -2.37% | -0.00030 |
| Noise/JPEG | +0.02455 | -0.00145 | +2.09% | 0.00000 |
| Clean | -1.4509 | -0.00090 | +69.48% | -0.00010 |
| Horizontal motion blur | +0.17535 | +0.00310 | -0.34% | -0.00010 |

Lower LPIPS is better. On clean images the absolute median LPIPS increase is
0.00225 from a near-zero baseline; the large relative percentage is not a useful
standalone quality claim. Noise/JPEG still has a perceptual regression. Every
condition has 12/12 ArcFace coverage; the worst paired delta is -0.0033.

The synthetic blurred-edge regression increases the maximum edge slope from
21 to 26 intensity levels per pixel, approximately 24%, without adding overshoot
or changing color differences. This is a controlled edge test, not a claim that
all photographs become 24% clearer. Saved output images were also visually
inspected; severe blur is still visibly present after sharpening.

These results support stronger source-edge sharpening, but **do not pass the
earlier broad restoration target of 5% LPIPS improvement or blinded A/B gate**.
The change addresses the user's renewed sharpening priority; it must not be
reported as completion of the full restoration-quality plan. Clean images can
be overprocessed and noisy images may look harsher; lower the strength for them.

Reproduce the fixed quality check with local validation data and metric weights:

```powershell
python tools/validate_sharpening.py --output artifacts/sharpening_check_new
```

Existing output directories are rejected. The report records selected paths,
source hashes, code hashes, versions and per-case metrics, and saves original,
old and new outputs at the same display size.

## CPU and verification

Warmed filter plus resize only, 8 OpenCV threads, three warmups and 21 alternating
samples per method, on the local Windows machine:

| Input → output | Old median / p95, ms | New median / p95, ms |
| --- | ---: | ---: |
| 128 → 512 | 38.24 / 43.85 | 31.75 / 36.06 |
| 256 → 512 | 46.58 / 51.09 | 41.80 / 56.68 |
| 512 → 1024 | 176.81 / 194.38 | 166.02 / 206.71 |
| 512 → 512 | 46.05 / 53.30 | 54.23 / 58.40 |

Latency is mixed: median resizing cases improve, but some p95 values and 1x
processing regress. This is not a universal speedup or the previous <=10%
latency gate. Peak RAM and Linux production latency have not been measured.
Evidence: `artifacts/enhance_quality_check/adaptive_cpu.json`.

- 23 reliability tests pass on Python 3.11 / OpenCV 4.10.
- Real pipeline integration passes: four-face AI restoration, new Pure branch,
  channel-difference preservation, bounded increments, non-face SR and discovery.
- Streamlit AppTest passes Wink/Natural/Pure switching, including new control
  labels and Pure 0.50 / Natural 0.35 defaults.
- Full fixed 60-case check completes with finite metrics and unchanged BGR
  channel differences relative to the Lanczos output for every case.

Remaining restoration work includes controlled AI detail fusion, realistic blur
handling, larger visual assessment and CPU/RAM release gates. Do not tune on the
fixed check set repeatedly to manufacture a passing score.
