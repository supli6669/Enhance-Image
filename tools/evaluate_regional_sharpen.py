"""Development-only experiment runner. Does not open the frozen check partition."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.regional_sharpen import analyze, sharpen


def degradation(reference, condition, seed):
    scale = 1 if condition == 'clean' else (4 if condition == 'severe' else 2)
    image = reference
    if condition.startswith('motion'):
        kernel = np.eye(7, dtype=np.float32)/7 if condition.endswith('diagonal') else np.zeros((7, 7), np.float32)
        if condition.endswith('_h'):
            kernel[3] = 1/7
        image = cv2.filter2D(image, -1, kernel)
    elif condition == 'anisotropic':
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=2, sigmaY=.5)
    elif condition != 'clean':
        image = cv2.GaussianBlur(image, (0, 0), 2.2 if condition == 'severe' else 1.2)
    image = cv2.resize(image, (512//scale, 512//scale), interpolation=cv2.INTER_AREA)
    if condition == 'noise_jpeg':
        image = np.clip(image.astype(float)+np.random.default_rng(seed).normal(0, 6, image.shape), 0, 255).astype(np.uint8)
        ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ok:
            raise RuntimeError('JPEG encode failed')
        image = cv2.imdecode(data, 1)
    return image, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--perceptual', action='store_true', help='LPIPS on development only at the selected strength.')
    parser.add_argument('--identity', action='store_true', help='Also run ArcFace for a perceptually useful candidate.')
    parser.add_argument('--variants', nargs='+', choices=('regional_noise', 'regional_structure', 'guided_structure'))
    parser.add_argument('--strength', type=float, choices=(.35, .5, .65), default=.5,
                        help='Strength for perceptual follow-up; pixel screening uses all frozen strengths.')
    args = parser.parse_args()
    manifest = json.loads((args.experiment/'manifest.json').read_text())
    spec = importlib.util.spec_from_file_location('frozen_wink', args.experiment/'wink_enhancer.py')
    baseline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_module)
    baseline = baseline_module.WinkQualityEnhancer()
    cv2.setNumThreads(8)
    evaluator = None
    if args.perceptual:
        import torch
        torch.set_num_threads(8)
        from tools.evaluate_restoration import BenchmarkEvaluator
        evaluator = BenchmarkEvaluator(device='cpu', enable_identity=args.identity)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'scope': 'development only; no promotion', 'samples': [],
              'manifest_sha256': hashlib.sha256((args.experiment/'manifest.json').read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256((ROOT/'tools/regional_sharpen.py').read_bytes()).hexdigest(),
              'baseline_sha256': hashlib.sha256((args.experiment/'wink_enhancer.py').read_bytes()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'variants': args.variants or manifest['variants'][1:], 'perceptual': args.perceptual,
              'identity': args.identity and args.perceptual, 'strength': args.strength if args.perceptual else manifest['strengths'],
              'timing_note': 'Unwarmed diagnostics; not a CPU gate.'}
    (args.output/'regional_sharpen.py').write_bytes((ROOT/'tools/regional_sharpen.py').read_bytes())
    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    save()
    try:
        for index, entry in enumerate(manifest['partitions']['development']):
            path = (ROOT/'models/CodeFormer/datasets/ffhq/ffhq_512'/entry['path']).resolve()
            if not path.is_relative_to((ROOT/'models/CodeFormer/datasets/ffhq/ffhq_512').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Dataset path or hash changed')
            reference = cv2.resize(cv2.imread(str(path)), (512, 512), interpolation=cv2.INTER_AREA)
            for condition in manifest['conditions']:
                image, scale = degradation(reference, condition, 13100+index)
                folder = args.output/f'{index}_{condition}'
                folder.mkdir()
                maps = analyze(image)
                for key in ('noise_confidence', 'detail_support', 'edge_protection'):
                    if not cv2.imwrite(str(folder/f'{key}.png'), np.rint(maps[key]*255).astype(np.uint8)):
                        raise OSError('Could not save map')
                outputs = {'baseline': baseline.apply_adaptive_sharpen(image, .5, upscale=scale)}
                for variant in (args.variants or manifest['variants'][1:]):
                    for strength in ([args.strength] if args.perceptual else manifest['strengths']):
                        outputs[f'{variant}_{strength}'] = sharpen(image, strength, scale, variant, maps)
                row = {'path': entry['path'], 'condition': condition, 'metrics': {}}
                for label, output in outputs.items():
                    metrics = evaluator.evaluate_pair(output, reference) if evaluator else {
                        'psnr': float(cv2.PSNR(reference, output)),
                        'ssim': float(structural_similarity(reference, output, channel_axis=2, data_range=255))}
                    row['metrics'][label] = metrics
                    if not cv2.imwrite(str(folder/f'{label}.png'), output):
                        raise OSError('Could not save image')
                report['samples'].append(row)
                save()
            print(f'Completed {index+1}/24 development references', flush=True)
        report['summary'] = {}
        for condition in manifest['conditions']:
            rows = [r for r in report['samples'] if r['condition'] == condition]
            result = {}
            for label in rows[0]['metrics']:
                metrics = {}
                for key in rows[0]['metrics'][label]:
                    values = [r['metrics'][label][key]-r['metrics']['baseline'][key] for r in rows if key in r['metrics'][label] and key in r['metrics']['baseline']]
                    metrics[key] = {'median_delta': float(np.median(values)) if values else None, 'coverage': len(values)}
                result[label] = metrics
            report['summary'][condition] = result
        report['status'] = 'completed'
        save()
        print(json.dumps(report['summary']), flush=True)
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        save()
        raise


if __name__ == '__main__':
    main()
