# Blur diagnosis follow-up — 2026-09-14

## Decision and evidence

Do not implement automatic deblurring from the current scalar edge-width feature.
The fixed deblur operator already failed quality gates. Routing would introduce
a second unverified component rather than repair that operator.

Ran tools/audit_blur_routing.py on completed development screening: leave one
source out, keeping all seven variants of that source out of threshold selection.
For each fold the threshold is the maximum edge width among the other 23 sources'
clean and noise/JPEG controls. Trigger requires strictly greater width; missing
features abstain. This deliberately conservative rule tests whether avoiding
control processing leaves useful blur coverage. No threshold grid was searched.

| Held-out condition | Triggered / 24 |
|---|---:|
| Clean | 0 |
| Noise/JPEG | 1 |
| Soft | 1 |
| Severe | 1 |
| Horizontal motion | 1 |
| Diagonal motion | 2 |
| Anisotropic | 1 |

Only 6/120 blurred cases trigger (5% coverage), and one noisy case still triggers.
This is development evidence against this particular rule, not proof that all
blur classifiers fail. The feature was previously observed; cross-validation is
not an independent final check. One clean sample has unavailable width. Grouping
prevents synthetic variants of the held-out reference entering its training fold;
identities across different source files have not been reviewed.

## Research implications

[Fang et al., CVPR Workshops 2022](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Fang_A_Robust_Non-Blind_Deblurring_Method_Using_Deep_Denoiser_Prior_CVPRW_2022_paper.html)
describe ringing from inaccurate supplied blur kernels. Their learned denoiser
method is research context, not an implemented or CPU-benchmarked dependency here.

[Gu et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gu_Blind_Super-Resolution_With_Iterative_Kernel_Correction_CVPR_2019_paper.pdf)
study kernel mismatch in blind super-resolution and iterative correction. This
supports investigating the blur/downsampling model jointly; it does not establish
that the model is suitable for this app's CPU budget or preserves identity.

Inference from our results: prioritize blur-model feasibility and realistic
controls before another global sharpening-strength sweep or a UI auto toggle.

## Next bounded experiment specification

1. Assemble a separately labeled real-blur development panel (camera shake,
   defocus, compression/noise, already sharp, mixed sharp/blur regions). Record
   source provenance and usage terms; do not silently repurpose the check set.
   Real images without references support visual assessment, not reference PSNR.
2. On synthetic development only, compare matched and deliberately mismatched
   Gaussian, directional-motion and anisotropic PSFs at native resolution first,
   then with the existing downsampling conditions. Known-kernel results are an
   explicitly labeled diagnostic upper bound, never production inference scores.
3. Require the matched operator to reach the existing 5% LPIPS blur gate with
   PSNR/SSIM constraints before investing in blind kernel estimation. Failure
   means change the restoration operator, not just its selector.
4. Only if that feasibility gate passes, evaluate input-only kernel estimates
   and uncertainty. Test wrong angles/lengths, noise, texture, and insufficient
   edges; abstain on insufficient evidence. No reference/condition input at runtime.
5. Preserve geometry/channel checks; inspect halos and texture at equal scale.
   Independent identity/A-B and deployment CPU/RAM checks still precede release.

App remains v3.0.2. No production routing, new model, or release claim is made.
