# Reliability and model-quality recovery — 2026-09-07

Approved scope: implement the priorities in the project audit. CPU inference is
the deployment target. Existing checkpoints and uncommitted Real-ESRGAN changes
must be preserved. No quality promotion based on toy training or loss alone.

1. [x] Repair imports, face masks, cached upscale state, explicit model routing,
   and share a parameter snapshot across photo/video/batch. Verify regression
   fixtures and existing integration suites on CPU.
2. [x] Remove embedded Kaggle credentials. Use environment credentials, stage
   downloads separately, reject unsafe paths, and inventory model graph/sidecars
   with hashes. Never replace active checkpoints automatically.
3. [x] Make training fail closed without pretrained teachers and real data.
   Enumerate images recursively with explicit domain selection and holdout
   exclusion. Add a preflight command that does not start training.
4. [x] Make evaluation select an explicit checkpoint and record provenance,
   median metrics and latency. Make INT8 calibration use representative real
   inputs, fail on missing data, and never silently substitute dynamic INT8.
5. [x] Run tests and a labeled evaluation smoke run; document actual results,
   remaining quality gates, and reproducible commands. Commit verified changes.

Production training and promotion gates: reviewed train/validation/test split,
full baseline and candidate reports on the same held-out data, human A/B identity
review, and acceptable CPU latency. A smoke evaluation is not that quality gate.
The compromised Kaggle credential must be revoked in the account; removing it
from the working tree does not revoke it or erase historical commits.

## Enhance optimization follow-up — 2026-09-12

User selected enhance quality diagnosis and research rather than Kaggle pilot
work. Evidence and next experiments: ENHANCE_OPTIMIZATION_REPORT.md.

- [x] Separate internal model fidelity and source pixel blending without changing
  defaults; validate endpoints, legacy behavior and invalid inputs.
- [x] Complete 3-reference / 9-case stage diagnostics and retain interrupted-run
  provenance. Reject tested stronger blends against the proposed quality limits.
- [x] Compare a real PTH case in an isolated Python 3.11 environment matching the
  main deployment dependency versions. Scope is Windows smoke validation only.
- [x] Test geometric one-pass warp separately on synthetic round trips. Keep
  production warp unchanged until real crop/mask/seam verification.
- [x] Defer UI CodeFormer allocation until needed; verify real lazy-to-AI
  integration, Pure behavior, and preset switching.
- [ ] Expand/freeze development and internal check sets, including clean,
  motion-blurred, small/occluded and multiple-face cases.
- [ ] Test parsing-masked detail fusion and actual affine/mask variants separately;
  accept only candidates meeting the recorded quality limits.
- [ ] Measure cold startup, warmed CPU median/p95 and peak RAM, then thread counts.
- [ ] Run fixed quality evaluation and blinded A/B before selecting a new default.

Training and final model promotion still follow PROJECT_IMPROVEMENT_PLAN.md.

## User priority: visible sharpening — 2026-09-13

The user explicitly rejected nearly unchanged outputs and reiterated the need
to sharpen images. The bounded implementation scope is source-scale adaptive
sharpening with a stronger Pure default; see SHARPENING_UPDATE.md.

- [x] Compare source-scale filters on development references and select one.
- [x] Implement noise-cored luminance detail before upscaling, with output halo
  constraints and no extra default face reconstruction.
- [x] Validate on 12 separate references / 60 cases; record both improvements
  and regressions. This does not pass the broader restoration-quality gate.
- [x] Verify synthetic edges, flat fields/noise, color preservation, real model
  integration, and warmed CPU timing.
- [ ] Complete broad restoration/A-B/production CPU and RAM gates described above.

## Detailed structure-preserving sharpening plan — 2026-09-13

User requested a detailed plan to sharpen images without deforming them.
See DETAIL_PRESERVING_SHARPEN_PLAN.md for P0–P6, data separation, regional maps,
conditional parsing/deblur, frozen acceptance criteria and runtime verification.
This is planning only. Begin with P0–P2; no new algorithm or default is changed
by this documentation update. Existing failed quality gates remain open.

P0–P2 experiment update: froze 24 development / 24 check / 55 reserve images,
excluding 19 previously observed references. Local noise/structure maps and
three experimental filters implemented outside production. Pixel screening and
LPIPS follow-up each completed 168 development cases; candidates failed the
perceptual gates. Five synthetic tests pass and are included in CPU CI. See
REGIONAL_SHARPEN_PROGRESS.md. Check-set evaluation and promotion remain pending;
do not claim P2's quality exit criterion passed. Coverage annotation remains open.

Conditional P4 update: implemented bounded source-only Gaussian inverse filtering
outside production. Four fixed configurations screened on 168 development cases;
LPIPS follow-up for sigma .9/reg .02 completed another 168 cases. Blur improvements
of .80–3.36% miss the 5% gate; clean LPIPS +.00575 and noise/JPEG regression reject
promotion. Six deblur tests and five regional tests pass. See
BOUNDED_DEBLUR_PROGRESS.md. Check data, identity/A-B and CPU/RAM gates remain open.
