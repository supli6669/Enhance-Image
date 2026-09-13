# Regional sharpening experiment — 2026-09-13

This continues P0–P2 of DETAIL_PRESERVING_SHARPEN_PLAN.md. The production
pipeline, UI settings and v3.0.2 sharpening implementation are unchanged.
The new functions live under tools and are not selected by the app.

## Frozen experiment

`tools/prepare_sharpen_experiment.py` created the private local artifact
`artifacts/regional_sharpen_v1/manifest.json`, with seed 9132026:

- 24 development references, 24 independent-check references, 55 reserve.
- 19 previously observed references excluded from development/check.
- Whole-image exact-pixel hashes and 63-bit DCT pHash suggestions checked;
  no pairs within the conservative distance threshold of 8 were found among
  the 122 validation references. This does not establish identity separation.
- Independent set-membership audit confirmed no partition/observed overlap.
- Existing experimental validation paths retain the previous quarantine policy;
  original train, validation and holdout manifests were not modified.
- Frozen baseline source and acceptance-plan copies are stored with the manifest.
  Baseline snapshot was taken at 1dbf1e9, whose sharpening math is v3.0.2.

Real images remain portrait data. Pose, glasses and small-face coverage have not
been manually labeled. General-image regression covers synthetic text, edges,
flat colors and mixed noise; it does not replace a natural scene/product set.
Data partition freezing is complete; the wider coverage audit remains open.

Seven fixed conditions: clean, soft blur, severe downsampling, noise/JPEG,
horizontal motion, diagonal motion and anisotropic blur. Each condition is
generated from the same reference/seed for all variants. The independent check
partition is not evaluated by the development runner.

## Experimental implementations

`tools/regional_sharpen.py` contains:

- Local noise estimates from diagonal high-pass residuals in approximately
  32-pixel tiles, selecting lower-gradient pixels, with interpolation/feathering.
- Noise-adjusted local variance for detail support and a fine/medium-energy
  heuristic to protect already-sharp/noisy regions. These are heuristics,
  not calibrated probabilities and not semantic face masks.
- `regional_noise`: same detail bands with regional noise coring/gain.
- `regional_structure`: also applies support and edge protection.
- `guided_structure`: self-guided box-filter decomposition plus the same maps.

Strengths 0.35/0.50/0.65 were fixed in the manifest. All variants apply one
shared luminance increment, with local extrema/headroom bounds before/after
resizing. No image warp, face synthesis, parsing or model training was added.

## Verification and reproducibility

The dedicated synthetic tests check mixed-noise localization, map ranges,
color preservation, new edge extrema, flat/disabled behavior, invalid maps,
text-versus-noise support and non-mutation of the source image. They are added
to CPU CI. They do not certify perceptual quality on photographs.

```powershell
python tools/test_regional_sharpen.py
python tools/prepare_sharpen_experiment.py --output artifacts/regional_sharpen_new
python tools/evaluate_regional_sharpen.py --experiment artifacts/regional_sharpen_new --output artifacts/regional_pixel_new
```

Output directories must be new. Development evaluation uses the frozen baseline
module and verifies source-image hashes. Outputs include maps and per-case scores.
The script deliberately provides no independent-check evaluation option yet.
The follow-up `--perceptual --strength 0.5` computes LPIPS on development only;
`--identity` adds ArcFace for a perceptually useful candidate. No identity run
was needed here because the perceptual stage failed first. Timings in diagnostic
work must not be treated as a warmed CPU benchmark.

Implementation changes between the pixel run and follow-up only added explicit
input validation and runner provenance/options; filter math did not change.
Each run retains its own regional source snapshot/hash. Existing Kaggle pilot
changes remain separate and must not be included in this experiment's commit.

## Release decision

Both development runs completed: pixel screening has 168 cases x 10 outputs
(baseline plus 3 variants x 3 strengths); perceptual follow-up has 168 cases x
3 outputs (baseline, regional_noise and regional_structure at strength 0.50).
Artifacts: development_pixel/report.json and development_lpips/report.json in
the frozen experiment directory. All LPIPS cells have 24/24 coverage per group.

Median LPIPS differences against v3.0.2 (negative is better):

| Condition | Regional noise, 0.50 | Regional structure, 0.50 |
| --- | ---: | ---: |
| Clean | -0.00035 | -0.00225 |
| Soft blur | +0.00065 | +0.00385 |
| Severe downsampling | -0.00170 | +0.00265 |
| Noise/JPEG | +0.00250 | +0.00090 |
| Horizontal motion | +0.00050 | +0.00265 |
| Diagonal motion | +0.00025 | +0.00205 |
| Anisotropic blur | +0.00065 | +0.00345 |

Neither variant meets the required perceptual improvement on mild blur or the
noise/JPEG gate. Regional structure reduces clean-image overprocessing (median
PSNR +4.014 dB), but loses mild-blur detail (PSNR -0.109 dB and worse LPIPS).
This is protection by reducing enhancement, not the requested increase in clarity.
Guided structure at 0.50 was rejected after pixel screening: mild-blur PSNR
-0.478 dB and horizontal-motion SSIM -0.00663 fail the recorded limits.

Decision: reject these candidates for default promotion. The noise/structure
maps remain experimental diagnostic tools. Five synthetic tests passed; these
tests establish basic invariants, not a restoration-quality pass. No ArcFace,
independent-check, A/B or deployment-performance gate is claimed as passed.
The 24-image check partition remains unevaluated and available for a genuinely
useful next candidate. Production source files are unchanged.

Next development hypothesis: the current candidates mostly attenuate the same
detail bands; they cannot restore the lost frequencies responsible for blur.
Before adding semantic masks, characterize edge-spread/blur on development data
and test bounded source-only deblurring as described in P4, with clean/noise and
incorrect-kernel controls. Keep v3.0.2 as the stable comparator. This is a proposed
next experiment, not a proven solution.

No candidate is promoted by adding these tools. Independent quality/A-B,
portrait coverage, CPU/RAM, integration and release gates remain required before
changing the application default. Freeze a candidate only after development
evidence shows a useful tradeoff; do not inspect the check set to choose one.
