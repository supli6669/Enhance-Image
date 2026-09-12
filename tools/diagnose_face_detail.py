"""Compare model fidelity and source blending without retraining or changing defaults.

Uses validation references only. A new output directory is required to preserve
past evidence. Each fidelity value runs the model once per input; blending reuses
the actual restored crops. This is development evidence, not a release gate.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from pipeline import LocalAIEnhancerPipeline
from tools.evaluate_restoration import BenchmarkEvaluator
from tools.model_artifacts import describe_model, sha256_file


def degraded(reference, condition, seed):
    scale = 4 if condition == 'severe' else 2
    sigma = 2.2 if condition == 'severe' else 1.2
    image = cv2.resize(cv2.GaussianBlur(reference, (0, 0), sigma),
                       (512 // scale, 512 // scale), interpolation=cv2.INTER_AREA)
    if condition == 'noise_jpeg':
        rng = np.random.default_rng(seed)
        image = np.clip(image.astype(float) + rng.normal(0, 6, image.shape), 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image, scale


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write {path}')


def summary(rows):
    result = {}
    for condition in sorted({row['condition'] for row in rows}):
        cases = [r for r in rows if r['condition'] == condition]
        modes = sorted({mode for row in cases for mode in row['outputs']})
        result[condition] = {}
        for mode in modes:
            values = [row['outputs'][mode]['metrics'] for row in cases if mode in row['outputs']]
            result[condition][mode] = {'count': len(values)}
            for key in ('psnr', 'ssim', 'lpips', 'identity_similarity'):
                available = [v[key] for v in values if key in v and np.isfinite(v[key])]
                result[condition][mode][key] = float(np.median(available)) if available else None
                result[condition][mode][key + '_coverage'] = len(available)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'models/CodeFormer/datasets/ffhq/ffhq_512')
    parser.add_argument('--validation', type=Path, default=ROOT / 'benchmarks/splits/portraits_experiment_v1/validation.txt')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--case-indices', type=int, nargs='+',
                        help='Run only these zero-based positions in the selected reference set; preserves seeds.')
    parser.add_argument('--fidelity', type=float, nargs='+', default=[0.5, 0.85])
    parser.add_argument('--blend', type=float, nargs='+', default=[0.0, 0.4, 0.85])
    parser.add_argument('--conditions', nargs='+', choices=['soft', 'noise_jpeg', 'severe'],
                        default=['soft', 'noise_jpeg', 'severe'])
    args = parser.parse_args()
    if args.limit < 1 or any(not np.isfinite(v) or not 0 <= v <= 1 for v in args.fidelity + args.blend):
        parser.error('limit must be positive; fidelity/blend must be finite in [0, 1]')
    names = [n.strip() for n in args.validation.read_text(encoding='utf-8').splitlines() if n.strip()]
    if args.limit > len(names):
        parser.error('limit exceeds validation reference count')
    if args.case_indices is not None and (len(set(args.case_indices)) != len(args.case_indices)
            or any(i < 0 or i >= args.limit for i in args.case_indices)):
        parser.error('case-indices must be distinct positions below limit')
    indices = np.linspace(0, len(names) - 1, args.limit, dtype=int)
    selected = [names[i] for i in indices]
    paths = [(args.dataset / name).resolve() for name in selected]
    if any(not p.is_relative_to(args.dataset.resolve()) or not p.is_file() for p in paths):
        parser.error('Validation contains missing or unsafe paths')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    torch.manual_seed(42)
    report = {'status': 'running', 'scope': 'validation diagnostics; no promotion',
        'environment': {'python': sys.version, 'platform': platform.platform(),
            **{name: importlib.metadata.version(name) for name in
               ('torch', 'torchvision', 'numpy', 'onnxruntime', 'scikit-image', 'lpips')},
            'opencv': cv2.__version__, 'threads': 8},
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_hashes': {name: sha256_file(ROOT / name) for name in
                          ('pipeline.py', 'wink_enhancer.py', 'tools/diagnose_face_detail.py')},
        'model': describe_model(args.model), 'validation_sha256': sha256_file(args.validation),
        'selected': selected, 'fidelity': args.fidelity, 'blend': args.blend,
        'conditions': args.conditions, 'case_indices': args.case_indices, 'samples': [],
        'timing_note': 'Unwarmed diagnostics, not a CPU benchmark. b0.85 timing is full inference; other blends time paste plus clarity only.'}
    for name in report['source_hashes']:
        snapshot = args.output / 'source_snapshot' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, snapshot)

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    try:
        pipe = LocalAIEnhancerPipeline(device='cpu', model_path_override=str(args.model))
        evaluator = BenchmarkEvaluator(device='cpu')
        for index, path in enumerate(paths):
            if args.case_indices is not None and index not in args.case_indices:
                continue
            reference = cv2.imread(str(path))
            if reference is None:
                raise ValueError(f'Unreadable reference: {path}')
            reference = cv2.resize(reference, (512, 512), interpolation=cv2.INTER_AREA)
            for condition in args.conditions:
                image, scale = degraded(reference, condition, 100 + index)
                case_dir = args.output / f'{index}_{condition}'
                case_dir.mkdir()
                write_image(case_dir / 'reference.png', reference)
                write_image(case_dir / 'input.png', image)
                base = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                row = {'path': selected[index], 'source_sha256': sha256_file(path),
                       'condition': condition, 'scale': scale, 'outputs': {}, 'stages': []}

                def record(label, output, latency=None):
                    if output.shape != reference.shape:
                        raise ValueError(f'Unexpected output shape {output.shape}')
                    write_image(case_dir / f'{label}.png', output)
                    row['outputs'][label] = {'metrics': evaluator.evaluate_pair(output, reference),
                                             'latency_ms': latency}

                record('Lanczos', base)
                record('Pure', pipe.process_image(image, upscale=scale, preset_mode='Pure Quality'))
                for fidelity in args.fidelity:
                    start = time.perf_counter()
                    full = pipe.process_image(image, upscale=scale, preset_mode='Custom',
                        w=fidelity, source_blend=0.85, enable_super_clarity=True,
                        clarity_strength=0.35, enable_dehaze=False, enable_deblur=False,
                        wink_mode=False, face_upsample=False)
                    elapsed = (time.perf_counter() - start) * 1000
                    helper = pipe._face_helper_cache['retinaface_mobile0.25']
                    stage = {'fidelity': fidelity, 'face_count': len(helper.restored_faces),
                             'full_inference_ms': elapsed, 'crops': []}
                    for face_index, (crop, restored, affine) in enumerate(zip(
                            helper.cropped_faces, helper.restored_faces, helper.affine_matrices)):
                        transform = affine.copy()
                        transform[:, :2] /= scale
                        aligned_ref = cv2.warpAffine(reference, transform, helper.face_size)
                        prefix = f'w{fidelity}_face{face_index}'
                        for label, crop_image in [('original', crop), ('raw_ai', restored), ('reference', aligned_ref)]:
                            write_image(case_dir / f'{prefix}_{label}.png', crop_image)
                        stage['crops'].append({'original': evaluator.evaluate_pair(crop, aligned_ref),
                                               'raw_ai': evaluator.evaluate_pair(restored, aligned_ref)})
                    row['stages'].append(stage)
                    # Include a fixed 85% source blend as the full pipeline control.
                    # This matches legacy coupling only when fidelity is also 0.85.
                    record(f'w{fidelity}_b0.85', full, elapsed)
                    if not helper.restored_faces:
                        continue  # Explicit face_count=0; do not label fallback as AI restoration.
                    for blend in args.blend:
                        if blend == 0.85:
                            continue
                        start = time.perf_counter()
                        pasted = pipe.paste_faces_custom_blend(helper, upscale=scale,
                            blend_softness=0.5, w=fidelity, source_blend=blend, wink_mode=False)
                        pasted = pipe.wink_enhancer.apply_laplacian_pyramid_clarity(pasted, strength=0.35 * 0.45)
                        paste_ms = (time.perf_counter() - start) * 1000
                        record(f'w{fidelity}_b{blend}', pasted, paste_ms)
                report['samples'].append(row)
                save()
                print(f"Completed {index + 1}/{len(paths)} {condition}: {len(row['outputs'])} outputs", flush=True)
        report['summary'] = summary(report['samples'])
        report['status'] = 'completed'
        save()
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        save()
        raise
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
