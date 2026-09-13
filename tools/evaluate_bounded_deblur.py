"""Bounded PSF experiments on development only, with clean/mismatched controls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.bounded_deblur import characterize, deblur
from tools.evaluate_regional_sharpen import degradation

CONFIGS = {f's{sigma}_r{reg}': {'sigma': sigma, 'regularization': reg, 'strength': .5}
           for sigma in (.6, .9) for reg in (.02, .06)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_metrics(metrics):
    """Represent exact-match PSNR explicitly, without invalid JSON or fake scores."""
    result = dict(metrics)
    if result.get('psnr') == float('inf'):
        result['psnr'] = None
        result['psnr_exact_match'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT/'artifacts/regional_sharpen_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--perceptual', action='store_true')
    parser.add_argument('--identity', action='store_true')
    parser.add_argument('--configs', choices=list(CONFIGS), nargs='+')
    args = parser.parse_args()
    if args.identity and not args.perceptual:
        parser.error('--identity requires --perceptual')
    manifest_path = args.experiment/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    selected = args.configs or list(CONFIGS)
    cv2.setNumThreads(8)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'scope': 'Development only, no automatic routing or release',
              'configs': {name: CONFIGS[name] for name in selected}, 'samples': [],
              'manifest_sha256': digest(manifest_path),
              'baseline_sha256': digest(args.experiment/'wink_enhancer.py'),
              'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'opencv': cv2.__version__, 'threads': 8},
              'perceptual': args.perceptual, 'identity': args.identity,
              'source_hashes': {name: digest(ROOT/name) for name in ('tools/bounded_deblur.py', 'tools/evaluate_bounded_deblur.py', 'tools/evaluate_regional_sharpen.py')},
              'note': 'Specified Gaussian kernels are applied to every condition, including clean/noise/motion controls. This is not an inferred or oracle kernel.'}
    for name in report['source_hashes']:
        (args.output/Path(name).name).write_bytes((ROOT/name).read_bytes())
    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    save()
    try:
        spec = importlib.util.spec_from_file_location('frozen_baseline', args.experiment/'wink_enhancer.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline = module.WinkQualityEnhancer()
        evaluator = None
        if args.perceptual:
            import torch
            torch.set_num_threads(8)
            from tools.evaluate_restoration import BenchmarkEvaluator
            evaluator = BenchmarkEvaluator(device='cpu', enable_identity=args.identity)
            report['environment']['torch'] = torch.__version__
        dataset = (ROOT/'models/CodeFormer/datasets/ffhq/ffhq_512').resolve()
        for index, entry in enumerate(manifest['partitions']['development']):
            path = (dataset/entry['path']).resolve()
            if not path.is_relative_to(dataset) or digest(path) != entry['sha256']:
                raise ValueError('Dataset path/hash differs from frozen manifest')
            reference = cv2.imread(str(path))
            if reference is None:
                raise ValueError(f'Unreadable source: {path}')
            reference = cv2.resize(reference, (512, 512), interpolation=cv2.INTER_AREA)
            for condition in manifest['conditions']:
                image, scale = degradation(reference, condition, 13100+index)
                row = {'path': entry['path'], 'condition': condition, 'profile': characterize(image), 'metrics': {}}
                folder = args.output/f'{index}_{condition}'
                folder.mkdir()
                base = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                outputs = {'lanczos': base, 'baseline': baseline.apply_adaptive_sharpen(image, .5, upscale=scale),
                           **{name: deblur(image, upscale=scale, **CONFIGS[name]) for name in selected}}
                for name, output in [('reference', reference), *outputs.items()]:
                    if not cv2.imwrite(str(folder/f'{name}.png'), output):
                        raise OSError('Could not save image')
                    if name == 'reference':
                        continue
                    if output.shape != reference.shape:
                        raise ValueError('Unexpected output dimensions')
                    delta = output.astype(int)-base.astype(int)
                    np.testing.assert_array_equal(delta[:, :, 0], delta[:, :, 1])
                    np.testing.assert_array_equal(delta[:, :, 1], delta[:, :, 2])
                    row['metrics'][name] = evaluator.evaluate_pair(output, reference) if evaluator else {
                        'psnr': float(cv2.PSNR(reference, output)),
                        'ssim': float(structural_similarity(reference, output, channel_axis=2, data_range=255))}
                    row['metrics'][name] = report_metrics(row['metrics'][name])
                    row['metrics'][name]['source_delta_mae'] = float(np.abs(delta).mean())
                report['samples'].append(row)
                save()
            print(f'Completed {index+1}/24 development references', flush=True)
        report['summary'] = {}
        for condition in manifest['conditions']:
            rows = [row for row in report['samples'] if row['condition'] == condition]
            result = {}
            for name in selected:
                result[name] = {}
                for key in rows[0]['metrics'][name]:
                    if key == 'psnr_exact_match':
                        continue
                    values = [r['metrics'][name][key]-r['metrics']['baseline'][key] for r in rows if r['metrics'][name].get(key) is not None and r['metrics']['baseline'].get(key) is not None]
                    result[name][key] = {'median_delta': float(np.median(values)) if values else None, 'min_delta': min(values) if values else None, 'max_delta': max(values) if values else None, 'coverage': len(values)}
                if args.perceptual:
                    ratios = [100*(r['metrics'][name]['lpips']/r['metrics']['baseline']['lpips']-1) for r in rows if r['metrics']['baseline']['lpips'] > 0]
                    result[name]['lpips_relative_median_percent'] = float(np.median(ratios)) if ratios else None
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
