# Reliability recovery verification — 2026-09-07

## Implemented and verified

- Corrected vendored BasicSR import ordering, parsing-mask dimensions, cached
  upscale state, explicit PyTorch/ONNX routing and no-face Lanczos behavior.
- Preserved input coordinates for small images: a 310 × 491 input now produces
  exactly 620 × 982 at 2×. Shared photo/video/batch arguments include effects,
  restoration mode and model selection; photo caching includes the input hash.
- Removed embedded Kaggle credentials and automatic active-weight replacement.
  Added staged downloads, graph/sidecar validation and model hashes.
- Required pretrained VQGAN/ArcFace teachers, recursive domain-aware loading,
  deterministic validation, correct [-1, 1] metric conversion and preflight gates.
- Added explicit model evaluation, deduplicated reference samples, metric medians,
  aligned ArcFace measurements and report provenance. INT8 calibration requires
  real inputs; export verifies numerical parity before recording success.

## Evidence from this workspace

| Check | Result |
| --- | --- |
| Final `tools/test_all.py` | 9/9 suites passed, exit 0 |
| Reliability regressions | 9 tests passed |
| Workflow guards | 8 tests passed |
| Real pipeline integration | Face and non-face processing passed, exit 0 |
| Preset/model-switch integration | Four presets and model switching passed, exit 0 |
| Streamlit AppTest | Initial render and preset switch passed |
| Training preflight | 2,479 train / 130 validation; 391 unique holdout images; zero exact-pixel overlap |
| Baseline ONNX export | Parity passed at fidelity 0, 0.5 and 1; maximum absolute error 0.000413 |

The split excluded one unreadable image without deleting it. Its review status
remains `pending_identity_and_near_duplicate_review`. The benchmark has 500 rows
but only 391 distinct decoded reference images.

Local evidence is in the ignored `benchmarks/reports/` directory: final suite log
`regression_final.log`, integration logs `integration.log` and `ab_ui.log`, UI log
`ui_smoke.log`, smoke reports and `v3_comparison_smoke.json`. The generated baseline
ONNX and parity manifest are local, ignored artifacts, not shipped model upgrades.

## Three-image smoke evaluation

These are **means on three images**, intended to verify evaluation execution.
They do not establish model quality or controlled CPU performance. Latencies were
measured while other verification processes were active.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ArcFace cosine ↑ |
| --- | ---: | ---: | ---: | ---: |
| Original baseline exported to ONNX | 30.04 | 0.7349 | 0.3848 | 0.9470 |
| Existing local v3 ONNX | 29.25 | 0.7288 | 0.4226 | 0.9354 |

The comparison reported `not_eligible`: limited scope and regressions in the
median quality metrics. Baseline remains the default. No new training run,
candidate quantization, or production model promotion was performed.

## Remaining work before model improvement

1. Revoke the exposed Kaggle key in the account and configure replacement
   credentials privately. Historical commits still contain the old key.
2. Review identity and near-duplicate separation in the prepared split.
3. Run the full baseline evaluation, then a fresh GPU run using pretrained
   teachers and the reviewed real dataset. The historical Kaggle run completed
   6,000 iterations on 30 toy images; it is not evidence of restoration quality.
4. Compare the candidate on the identical full benchmark, review identity
   preservation visually, measure CPU latency under controlled conditions, and
   only then approve export/INT8 and deployment.

Pre-existing changes inside the Real-ESRGAN submodule are preserved and excluded
from this recovery commit. Local validation does not imply remote CI or a hosted
deployment has completed; those statuses must be checked after pushing.
