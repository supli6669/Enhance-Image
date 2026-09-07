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
