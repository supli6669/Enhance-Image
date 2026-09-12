---
title: Custom AI Face Enhancer
emoji: ✨
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: apache-2.0
---

# Custom AI Face Enhancer & Restorer

A self-hosted, local AI image enhancement web application built using **Python, Streamlit, PyTorch, and OpenCV**. It integrates the state-of-the-art **CodeFormer** model locally for face restoration with custom blending controls.

This project is configured to run out-of-the-box both **locally** on your CPU/GPU and **hosted on Hugging Face Spaces**.

---

## Key Features

- **Local Execution:** Runs directly on local CPU or NVIDIA GPU (via CUDA) for maximum privacy and processing speed.
- **Fidelity Weight Tuning ($w$):** Control the balance between generating rich realistic details (low $w$) and keeping high resemblance to the original face (high $w$).
- **Custom Blending Softness:** Exposes an adjustable soft mask feathering parameter to ensure smooth, seamless pasting of restored faces back into the upscaled background image.
- **Multiple Face Detectors:** Choose between highly accurate detectors (RetinaFace) or faster detectors for groups (YOLOv5).
- **Dark Mode UI:** Designed with custom glassmorphism and modern Outfit typography.

## Source-preserving enhancement (2026-09-12)

The UI's Pure Quality and Natural Likeness presets now skip face reconstruction,
beauty effects, dehaze and assumed-kernel deconvolution by default. They use one
bounded luminance clarity pass with Lanczos resizing. Clarity preserves channel
differences, suppresses tiny fluctuations and limits edge overshoot; dehaze now
scales its complete contrast effect with the strength control. Direct pipeline
calls default to `w=0.85`, `wink_mode=False`, and `enable_dehaze=False`; face
reconstruction remains available and enabled for those direct calls.

These defaults favor resemblance and modest sharpening of existing detail.
They cannot recover detail absent from the input. Restart the app to reload the
image-processing modules, then reselect a preset; saved custom presets retain
their previous settings. Run `python tools/test_reliability.py` for the filter
and preset regressions, and `python tools/test_pipeline.py` for model integration.

## Reliability recovery (2026-09-07)

The runtime now honors model selection and shares processing settings across
photo, video and batch. Lanczos remains the default CPU upscaler. Fidelity and
image heuristics are controls/measurements, not guarantees of facial identity.
See [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) for verified results and
remaining quality gates.

The default checkpoint is the original `weights/CodeFormer/codeformer.pth`.
A local `codeformer_baseline.onnx` is preferred only when its manifest verifies
source/output hashes and numerical parity. Export it explicitly:

```bash
python tools/export_onnx.py --checkpoint weights/CodeFormer/codeformer.pth --output weights/CodeFormer/codeformer_baseline.onnx
```

Exports refuse to overwrite existing models. Local candidate models are marked
unvalidated; malformed graphs and mismatched sidecars are excluded. Generated
ONNX files are not included in the Docker build, so the hosted default remains
PyTorch unless deployment packaging is explicitly updated.

Run the complete local checks (pretrained weights required), or the two offline
regression suites used in CI:

```bash
python tools/test_all.py
python tools/test_reliability.py
python tools/test_workflow_guards.py
```

### Reproducible data and evaluation

Create a new split directory from the real `faces` and `pinterest` domains.
Exact decoded-pixel duplicates and benchmark images are excluded. Unreadable
images cause an error unless explicitly recorded as exclusions:

```bash
python tools/prepare_training_split.py --output benchmarks/splits/real_portraits_v1 --exclude-unreadable
python train_custom.py --preflight --dataset-dir models/CodeFormer/datasets/ffhq/ffhq_512 --split-dir benchmarks/splits/real_portraits_v1
python tools/evaluate_restoration.py --model weights/CodeFormer/codeformer.pth --output-json benchmarks/reports/baseline_full.json
```

Preflight verifies data hashes, disjoint splits and required pretrained weights;
it does not train or approve a split. Use the review workflow below to freeze a
split with a review receipt. Editing `review_status` alone cannot pass the gate.
Production training also requires a full baseline report covering the same held-out benchmark:

```bash
python train_custom.py --fresh --dataset-dir models/CodeFormer/datasets/ffhq/ffhq_512 --split-dir benchmarks/splits/real_portraits_reviewed_v1 --baseline-report benchmarks/reports/baseline_reviewed_v1/report.json
```

Run training on a GPU environment with the same data/split/weights. Do not resume
the old toy-data run. For Kaggle, provide explicit dataset sources and the notebook
environment paths `TRAINING_DATA_DIR`, `TRAINING_SPLIT_DIR`, `BASELINE_REPORT` and,
after checkpoint review, `APPROVED_CHECKPOINT`.

Use the same evaluator settings for a candidate and compare reports:

```bash
python tools/evaluate_restoration.py --model artifacts/candidate.onnx --output-json benchmarks/reports/candidate_full.json
python tools/compare_evaluations.py --baseline benchmarks/reports/baseline_full.json --candidate benchmarks/reports/candidate_full.json --output benchmarks/reports/comparison.json
python tools/quantize_onnx_static.py --model artifacts/candidate.onnx --output artifacts/candidate_int8.onnx --calib-dir models/CodeFormer/datasets/ffhq/ffhq_512 --calib-manifest benchmarks/splits/real_portraits_v1/train.txt --max-samples 100
```

`--limit 3` creates a smoke evaluation, which cannot satisfy the production gate.
LPIPS/ArcFace dependencies fail explicitly; `--no-perceptual` records missing
metrics and cannot satisfy that gate. Reports record model hashes, sample hashes,
metric medians and latency. Static INT8 uses real calibration inputs and requires
a separate quality comparison with FP32; there is no automatic promotion.

### Credentials and artifacts

Kaggle tools read `KAGGLE_USERNAME` and `KAGGLE_KEY` from the environment. A key
previously committed in this repository must be revoked in the Kaggle account;
removing it from source does not revoke it or remove Git history. Never commit
replacement credentials. Downloads are staged under `artifacts/`, preserve
partial downloads and existing weights, and require review before activation.
Private splits, generated reports and staged artifacts are Git/Docker ignored.
GitHub CPU checks gate the automatic Hugging Face sync workflow.

### Human review and frozen baseline

For the lighter experimental workflow, use the separate local UI's **Duyệt nhanh**
mode. It shows optional samples and a short exception list instead of requiring
thousands of decisions. The tool conservatively quarantines non-holdout endpoints
of existing cross-split candidate pairs without making identity judgments:

```bash
python tools/prepare_experimental_split.py --output benchmarks/splits/portraits_experiment_v1
python tools/prepare_baseline.py --experimental --split-dir benchmarks/splits/portraits_experiment_v1 --output benchmarks/reports/experiment_baseline_run_v1 --execute
```

Original images and holdout are preserved. The split and reports are explicitly
experimental; unflagged identity overlap may remain. These reports cannot satisfy
production training or promotion gates. See DATASET_REVIEW_WORKFLOW.md for the
prepared local split, samples and evidence. The full manual-review path below
remains available separately.

```bash
python tools/review_dataset.py generate --split-dir benchmarks/splits/real_portraits_v1 --output benchmarks/reports/dataset_review_v1
```

Open `index.html` in the generated directory. All thumbnails and decisions remain
local. ArcFace scores and perceptual hashes suggest pairs; they do not identify
a person or certify that the rest of the dataset is free of leakage. Review every
individual image as well as pairs. Multi-face, small-face and undetected-face
images are flagged for review. `--skip-identity` creates a preview that cannot freeze.
The scan saves each completed feature atomically. If interrupted during feature
extraction, repeat the same command with `--resume`. Input hashes, configuration,
ArcFace weights and scanner code must match. Completed review directories cannot
be resumed or overwritten. Feature vectors are private local artifacts too.

For an interactive local reviewer with pending/cross-split filters and explicit
per-item saves, run the separate review UI (never deploy this private-data UI):

```bash
streamlit run tools/dataset_review_app.py --server.address 127.0.0.1 --server.port 8502
```

Edit only `decision` and `reviewer` in `decisions.csv`, preserving row order:

- Individual rows: `keep` or `exclude`.
- Pairs: `distinct`, `same_group`, `exclude_left`, `exclude_right`, `exclude_both`.
- Unresolved rows remain `pending`; they block freezing.

`same_group` links are transitive. Groups spanning splits retain holdout members
first, otherwise validation members, and exclude lower-priority members from the
new manifests. Original files are preserved. Excluding a holdout image requires a
separate benchmark revision; this tool rejects it. Groups missed by suggestions
must still be reviewed manually; exclude implicated training images if uncertain.

After review, create a new split and start its full baseline:

```bash
python tools/review_dataset.py freeze --split-dir benchmarks/splits/real_portraits_v1 --review-dir benchmarks/reports/dataset_review_v1 --reviewer "REVIEWER_NAME" --output benchmarks/splits/real_portraits_reviewed_v1
python tools/prepare_baseline.py --split-dir benchmarks/splits/real_portraits_reviewed_v1 --output benchmarks/reports/baseline_reviewed_v1 --execute
```

These commands refuse existing output directories. The baseline command checks
the review receipt, rehashes the data, confirms the holdout matches benchmark
reference pixels and records model/code/data provenance in `run.json`. Omit
`--execute` to prepare a run manifest only; use a new output directory when later
executing. Quality-evaluation latency is not a substitute for a controlled CPU
benchmark. Keep original review evidence with the frozen split.

---

## Local Execution Instructions

### Prerequisites
- Python 3.11
- Git

### Setup
1. Clone this repository to your local machine.
2. Initialize the virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # On Windows (PowerShell/CMD)
   source .venv/bin/activate   # On Linux/macOS
   ```
3. Run the custom BasicSR package patching script:
   ```bash
   python tools/patch_and_install_basicsr.py
   ```
4. Install the remaining requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Start the Streamlit server locally:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The app will automatically download the pretrained model weights on its first run.

---

## Hugging Face Spaces Deployment

This repository deploys as a **Docker Space**, not a Streamlit SDK Space. The
Dockerfile installs the pinned Python 3.11 CPU runtime and starts Streamlit on
port `7860`.

1. Log in to [Hugging Face](https://huggingface.co/) and create a new Space.
2. Set the Space SDK to **Docker** and select a CPU hardware tier appropriate
   for CodeFormer inference.
3. Push this repository to the Space. Keep Git LFS enabled: the tracked
   `weights/CodeFormer/codeformer.pth` model is required at build/runtime.
4. Wait for the Space build to complete, then open the Space URL. Additional
   optional model files are downloaded by the application only if unavailable.

### GitHub Actions sync

The included workflow syncs `main` to the configured Hugging Face Space and
uploads Git LFS objects first. Add a Hugging Face **write** token as the GitHub
Actions secret `HF_TOKEN`; do not place a token in a Git remote URL or commit it
to the repository. The sync intentionally does not force-push, so resolve any
divergent Space changes before running it again.
