# Enhance quality: findings and next steps (2026-09-12)

## Decision

Keep the source-preserving UI defaults. More AI contribution improves perceptual
distance in this small diagnostic set, but fails the proposed distortion and
facial-similarity limits. There is no validated replacement preset yet.

Two independently useful changes are implemented: an optional `source_blend`
parameter for controlled experiments, and lazy CodeFormer allocation in the UI.
These do not select a stronger restoration preset or change default image math.

## E1: Separate fidelity from source blending

Previously, `w` controlled both CodeFormer's internal fidelity and the original
pixel contribution when pasting. With `w=0.85`, the pasted crop contains 85%
source pixels and 15% AI pixels before the soft mask. The new API keeps this
behavior when `source_blend=None`; explicit values independently control source
contribution (0 = AI crop, 1 = source crop). Model fidelity still uses `w`.

The diagnostic harness used 3 validation references, each with soft blur,
noise/JPEG, and severe blur/downsampling: 9 cases, not 9 independent references.
It tested fidelity 0.5/0.85 and source blend 0/0.4/0.85, alongside Lanczos and
Pure. All outputs were 512 pixels square. The baseline ONNX model was fixed;
Wink, dehaze, deconvolution and neural face upscaling were disabled.

Median paired differences against Pure across 9 cases, at fidelity 0.85:

| Source contribution | LPIPS relative change | PSNR change | SSIM change | ArcFace change | Worst ArcFace change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 85% | -4.12% | -0.7603 dB | -0.0135 | -0.0017 | -0.0196 |
| 40% | -22.13% | -1.4150 dB | -0.0261 | -0.0102 | -0.0791 |
| 0% | -16.15% | -2.8090 dB | -0.0581 | -0.0309 | -0.1351 |

Lower LPIPS is better; higher PSNR/SSIM/ArcFace is better. Full-image ArcFace
coverage is 9/9 for every mode and measures the largest reference face, using
reference landmarks for both images. It does not certify every face in a group.

For 40% source contribution, median relative LPIPS changes by group were -20.45%
(soft), -22.13% (noise/JPEG), and -25.55% (severe). ArcFace median declined by
0.0102 in each group. The worst severe case was the reference containing several
people; the restoration detector found 1/3/2 faces across its degradation groups.
This needs explicit per-face review, not interpretation as a single-person case.

On the three soft cases, raw AI crops at fidelity 0.85 already had median LPIPS
0.0258 higher and ArcFace 0.0272 lower than the aligned source crops. Thus source
blending is not the sole limit. Across all groups there were 12 detected crops;
crop-level aggregates weight images with multiple detections more heavily.

The proposed candidate limits were at least 5% median LPIPS improvement, median
PSNR loss at most 0.2 dB, SSIM loss at most 0.005, and ArcFace loss at most 0.005.
None of the three rows above meets all limits. These exploratory data cannot
replace the planned larger fixed evaluation or blinded visual review.

Local evidence: `artifacts/enhance_quality_check/e1_combined.json`. The first run
was interrupted after 7 complete cases; a second completed the remaining 2.
Original reports and source snapshots remain intact in `e1_blend_local` and
`e1_blend_remaining`. Their model/config/environment match; source hashes differ
because lazy-loading support and diagnostic case selection were added between
runs. Lazy loading was not enabled in either diagnostic run. Timings are
unwarmed; full inference and paste-only variants must not be compared for speed.

## E0: Environment check

An isolated Windows environment ran a real PTH inference with Python 3.11.9,
PyTorch 2.3.1+cpu, NumPy 1.26.4, OpenCV 4.10.0, and ORT 1.18.1. The comparison
environment recorded Python 3.13.7, PyTorch 2.12.1+cpu, NumPy 2.4.6, OpenCV 5.0.0,
and ORT 1.27.0. Read versions from each run, not from earlier research notes.

Same reference, degradation and PTH SHA-256 on one soft case:

- Input, reference, Lanczos and Pure were pixel-identical across environments.
- AI output mean absolute difference was 0.06392/255; maximum was 3/255.
- AI PSNR was 30.6508 vs 30.6505; LPIPS was 0.1688 vs 0.1685.
- Raw AI crop maximum difference was 11/255, so intermediate outputs are not
  bit-identical even when the final source-heavy blend is close.

Both environments used eager loading. Source snapshots differ only in the
lazy-loading implementation on the pipeline side; the compared inference path
kept eager behavior. This is a smoke comparison, not causal isolation of a
specific dependency or proof of Linux deployment parity. Facexlib also installs
the non-headless OpenCV distribution; the actual imported target version was
4.10.0. Requirements and the main environment were not changed.

Evidence: `e0_local/report.json`, `e0_target/report.json`, `e0_comparison.json`
under the same local artifact directory.

## E2: Geometry experiment

A synthetic round-trip experiment tested rotation 0/5/15 degrees at output
scales 1/2/4. A composed single cubic warp improved PSNR over resize-Lanczos
then linear warp in all 9 geometric cases. At 5 degrees and 2x, PSNR was
48.1963 vs 47.2856 dB. This does not measure model restoration quality or speed.

Composition must include resize pixel-center translation `(scale - 1) / 2`.
Simply scaling the inverse affine omits this translation. Production geometry
and Lanczos defaults remain unchanged pending actual face/mask/seam validation.
Evidence: `e2_warp/report.json` and `warp_experiment.py` in the local artifacts.

## E3: Lazy allocation

`LocalAIEnhancerPipeline(lazy_load=True)` defers the CodeFormer model/session
until restoration requests it. The Streamlit resource now uses this option.
Pure processing skips model allocation; model selection and validation continue
to work. Constructor default remains eager for existing external callers.

This reduces unnecessary allocation for source-preserving requests. It does not
make model inference faster, and the first AI request still pays model load
cost. No percentage startup, RAM or throughput improvement has been established.

## Research-backed order of work

1. Expand development data before choosing another blend. Add clean controls,
   motion blur, small/occluded faces and multiple faces. Freeze a separate
   internal check set; retain the experimental label and the untouched final
   benchmark. Use the same source images/seeds across variants.
2. Test source-preserving detail fusion through existing parsing masks, initially
   on saved AI crops. Keep source low frequencies and geometry; limit AI detail
   near eyes/mouth, occlusion and face borders. This is a hypothesis to reject
   if it repeats the identity loss, not an assumed solution.
3. Validate composed warp separately with actual crops, masks, overlap order,
   odd dimensions and scales 1/2/4. Compare seams and ringing before combining
   it with fusion. Preserve the current default until those checks pass.
4. Profile cold loading separately from warmed detector/model/parsing/paste.
   Try ORT intra-op threads 1/2/4/8, sequentially on the same CPU, recording
   median/p95 and peak RAM. Keep cache/request locking; do not assume an outer
   thread pool parallelizes the already-locked ONNX calls.
5. If raw restoration still limits quality, benchmark a specialized denoiser or
   deblurrer against the existing model before training. NAFNet and compact
   Real-ESRGAN general-x4v3 are candidates, with measured CPU cost required.
   GPU diffusion models are not a justified CPU default. Resume training only
   through the existing recovery/pilot gates; do not revive old toy checkpoints.
6. Quantize only a candidate that passes quality. Calibrate on representative
   aligned training crops and supported fidelities, excluding evaluation data;
   keep sensitive operators in floating point if necessary.

Primary sources behind these choices:

- [CodeFormer implementation](https://github.com/sczhou/CodeFormer/blob/master/inference_codeformer.py): internal fidelity and restoration/pasting stages.
- [OpenCV affine transforms](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html) and [4-to-5 migration](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration): interpolation and version checks.
- [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html): measure thread configuration against the actual graph.
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html): accuracy diagnosis and selective quantization.
- [NAFNet official repository](https://github.com/megvii-research/NAFNet) and [Real-ESRGAN official repository](https://github.com/xinntao/Real-ESRGAN): conditional model candidates.
- [Real-ESRGAN paper](https://arxiv.org/abs/2107.10833): realistic combined degradation rather than blur alone.
- [Perception-distortion tradeoff](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html): perceptual gains do not guarantee lower distortion.

Detailed original research and the full proposed quality/A-B/CPU gates are in
the local `OPTIMIZATION_RESEARCH.md` and `RESOLUTION_PLAN.md` artifacts. Next
promotion still requires fixed evaluation, at least 50 blinded A/B cases with
at least 70% wins including preserved facial structure, and CPU/RAM checks.

## Verification

- Local and isolated Python 3.11 reliability suites: 18 tests passed in each,
  including independent blend control,
  legacy equivalence, invalid values, and deferred PTH/ONNX loading.
- Real pipeline integration: exit 0; lazy startup followed by four-face AI
  restoration, bounded/color-preserving Pure output, non-face SR and discovery.
- Streamlit AppTest: Wink, Natural and Pure preset switches passed; Natural/Pure
  keep reconstruction and aggressive beauty/dehaze/deblur controls disabled.
- E0 target inference and both completed E1 portions: successful output records;
  original interrupted-run status is retained rather than rewritten.

No model weights, training work, default fidelity, or default warp were changed.
