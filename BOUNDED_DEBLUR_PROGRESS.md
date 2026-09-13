# Bounded source-only deblurring experiment

Continues the conditional P4 investigation after the regional sharpening
candidates failed to improve blurred development images. v3.0.2 remains the
stable comparator. No production pipeline, model or preset is changed.

## Hypothesis and fixed scope

Attenuating existing detail bands protected clean images but did not restore
blurred edges. This experiment tests a regularized inverse Gaussian filter on
source luminance. It does not synthesize faces or move image coordinates.

Four predeclared configurations: sigma 0.6/0.9 crossed with regularization
0.02/0.06, all at strength 0.50. The inverse frequency response is capped at 2;
the final shared luminance increment is capped at 16 intensity levels. Local
extrema/headroom bounds are applied before and after resize. Reflect padding
prevents periodic DFT boundaries from wrapping across the image.

All four kernels are tested on every development condition, including clean,
noise/JPEG, anisotropic and horizontal/diagonal motion. Thus this is a
specified-kernel operator experiment with deliberate mismatches, not an oracle
using the known synthetic kernel and not an automatic blur classifier.

`characterize` measures monotonic edge-profile width and high-pass noise. It
records the number of accepted profiles and returns unavailable width when no
valid edge exists. Profiles can be correlated along one contour. These outputs
are diagnostic features, not confidence probabilities or a validated router.

## Data and environment

Reuse the frozen 24 development references and seven conditions from
artifacts/regional_sharpen_v1/manifest.json: 168 cases. The independent 24-image
check remains unevaluated. Source hashes are checked before processing.

The prior isolated Python 3.11 environment is no longer present in the workspace.
This experiment uses local Python 3.13.7, NumPy 2.4.6 and OpenCV 5.0.0 on Windows,
with eight OpenCV threads. Both baseline and candidate are rerun in this same
environment; scores are not mixed with earlier OpenCV 4.10 outputs. A candidate
would still need verification in the deployment environment before release.

Outputs use new directories under artifacts/bounded_deblur_v1. Each report
records the frozen manifest/baseline hashes, source snapshots, versions, config,
per-image profile and per-case scores. Images are saved at equal display size.

## Verification and commands

Six tests passed: exact-match PSNR JSON serialization, matched-blur PSNR improvement and stable edge
location; bounded wrong-kernel output and preserved channel differences; clean
step/flat/disabled behavior; invalid controls; no wraparound from a boundary
impulse. These are added to CPU CI. Passing them does not establish restoration
quality on real photographs or prove every mismatched kernel is harmless.

```powershell
python tools/test_bounded_deblur.py
python tools/evaluate_bounded_deblur.py --output artifacts/deblur_screening_new
```

For a configuration that survives pixel screening, use `--perceptual --configs`
with its recorded name to measure LPIPS; `--identity` adds ArcFace if needed.
Directories are never overwritten. The runner only evaluates development data.

The acceptance criteria remain those in DETAIL_PRESERVING_SHARPEN_PLAN.md.
No automatic application or default promotion follows from implementing this
operator. A useful operator would still need calibrated application conditions,
independent quality/visual assessment and CPU/RAM verification.

## Completed pixel screening

All 168 development cases completed for all four configurations. Median paired
PSNR changes in dB against the frozen baseline, rerun in the same environment:

| Condition | sigma .6 / reg .02 | .6 / .06 | .9 / .02 | .9 / .06 |
|---|---:|---:|---:|---:|
| Clean | +1.372 | +2.294 | -1.862 | -1.278 |
| Soft | -.030 | -.081 | +.130 | +.024 |
| Severe | +.012 | -.026 | +.095 | +.020 |
| Noise/JPEG | -.056 | -.059 | -.007 | +.008 |
| Horizontal motion | -.215 | -.229 | -.041 | -.053 |
| Diagonal motion | -.224 | -.231 | -.113 | -.124 |
| Anisotropic | -.171 | -.206 | +.014 | -.055 |

No static configuration passes every condition. Sigma .9/reg .02 is followed
up with LPIPS only to assess potential as a conditional blur operator: it has
already failed the clean-image gate, and noise/JPEG SSIM falls by .001943.
The weaker kernels fail the -.2 dB motion gate. No threshold is relaxed.

Edge-width ranges overlap: clean .775–1.378, soft 1.019–1.742, horizontal
motion .995–1.728. Thus a simple width threshold does not establish reliable
automatic routing. Noise estimates are also confounded by JPEG and texture.

The initial perceptual run stopped on the first exact-match Lanczos control:
PSNR is mathematically infinite, which strict JSON rejects. The corrected
runner stores that value as null with psnr_exact_match=true, excludes unavailable
paired values with explicit coverage, and labels summary extrema min/max.
The original failed directory is retained; perceptual_v2 is a fresh run.

## Next decision gates

Decision: **reject this fixed Gaussian inverse operator for release**.
perceptual_v2 completed all 168 cases, identity disabled. Median paired LPIPS
relative changes (negative is better): soft -3.36%, severe -2.13%, horizontal
motion -.80%, diagonal motion -1.57%, anisotropic -1.12%, noise/JPEG +.72%.
Clean absolute LPIPS increases .00575, exceeding the .001 gate; its relative
increase of 106.54% is inflated by the small baseline distance. None of the
blur groups reaches 5%. No ArcFace, independent check, blinded A/B or CPU/RAM
release gate is claimed passed. Six bounded-deblur and five regional tests pass.

The next research step must address kernel mismatch or spatially varying blur,
with an independently specified experiment and real-blur controls. Increasing
this operator's strength or routing by the synthetic condition is not supported
by these results. P4 remains an investigation, not an achieved quality milestone.

1. Complete paired LPIPS for sigma .9/reg .02 on all 168 development cases.
   If blur improvement is below 5%, reject this fixed operator before spending
   the check set or implementing a router that merely hides its regressions.
2. If the operator passes blur quality, collect/label independent clean, blur
   and noise controls before calibrating any routing. A synthetic condition
   label or reference image must never enter inference. Preserve the check set.
3. A future candidate must pass clean/noise preservation, identity and blinded
   same-size/crop comparisons, followed by production CPU/RAM gates. Portrait
   results alone do not establish performance on text, landscape or documents.
4. Only then connect a candidate to a preview/preset and release. Until then,
   production stays at v3.0.2; experiment commits do not warrant a version bump.
