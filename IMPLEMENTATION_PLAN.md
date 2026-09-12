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
