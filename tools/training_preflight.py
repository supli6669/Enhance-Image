"""Validate production training inputs before constructing a model or GPU job."""
from __future__ import annotations
import copy
import json
from pathlib import Path


def validate_quality_gate(split_report, baseline_path):
    from tools.prepare_baseline import require_reviewed_split
    require_reviewed_split(split_report)
    if split_report.get('review_status') != 'approved':
        raise ValueError('Review identity and near-duplicate separation, then approve the split before production training')
    if baseline_path is None:
        raise ValueError('A full --baseline-report is required before production training')
    report = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    summary = report.get('summary', {})
    if summary.get('evaluation', {}).get('data_review') == 'experimental':
        raise ValueError('Experimental baseline does not satisfy the production training gate')
    if summary.get('evaluation', {}).get('scope') != 'full':
        raise ValueError('Smoke evaluation cannot satisfy the baseline gate')
    if summary.get('total_samples', 0) < 300 or summary.get('total_samples') != summary.get('unique_reference_count'):
        raise ValueError('Baseline must cover all unique reference images (at least 300)')
    overall = summary.get('overall', {})
    for key in ('median_psnr', 'median_ssim', 'median_lpips', 'median_identity_similarity'):
        if overall.get(key) is None:
            raise ValueError(f'Baseline is missing required metric {key}')
    if not split_report.get('benchmark_manifest_sha256') or split_report['benchmark_manifest_sha256'] != summary.get('manifest_sha256'):
        raise ValueError('Baseline uses a different benchmark manifest')


def configure_training(config, dataset_root, split_dir, holdout):
    from tools.prepare_training_split import pixel_digest
    config = copy.deepcopy(config)
    dataset_root, split_dir, holdout = map(lambda p: Path(p).resolve(), (dataset_root, split_dir, holdout))
    split = json.loads((split_dir / 'split.json').read_text(encoding='utf-8'))
    groups = {}
    for name in ('train', 'validation'):
        entries = split[name]
        names = [line for line in (split_dir / f'{name}.txt').read_text(encoding='utf-8').splitlines() if line]
        if names != [entry['path'] for entry in entries]:
            raise ValueError(f'{name} manifest does not match recorded split')
        hashes = set()
        for entry in entries:
            path = (dataset_root / entry['path']).resolve()
            if not path.is_relative_to(dataset_root):
                raise ValueError('Split path escapes dataset root')
            digest = pixel_digest(path)
            if digest != entry['pixels_sha256']:
                raise ValueError(f'Dataset changed since split: {entry["path"]}')
            if digest in hashes:
                raise ValueError(f'Duplicate image in {name}')
            hashes.add(digest)
        groups[name] = hashes
    heldout = set()
    for relative in holdout.read_text(encoding='utf-8').splitlines():
        if relative.strip():
            path = (dataset_root / relative.strip()).resolve()
            if not path.is_relative_to(dataset_root):
                raise ValueError('Holdout path escapes dataset root')
            heldout.add(pixel_digest(path))
    if groups['train'] & groups['validation'] or (groups['train'] | groups['validation']) & heldout:
        raise ValueError('Data leakage across train, validation, or benchmark')
    if len(groups['train']) < 1000 or len(groups['validation']) < 20 or not heldout:
        raise ValueError('Production training requires >=1000 train, >=20 validation and nonempty holdout images')
    root = Path(__file__).resolve().parents[1]
    teacher = root / 'weights/facelib/vqgan_code1024.pth'
    identity = root / 'weights/facelib/recognition_arcface_ir_se50.pth'
    baseline = root / 'weights/CodeFormer/codeformer.pth'
    for path in (teacher, identity, baseline):
        if not path.is_file() or path.stat().st_size < 1_000_000:
            raise FileNotFoundError(f'Required pretrained weights missing: {path.name}')
    config['network_vqgan']['model_path'] = str(teacher)
    train = config['datasets']['train']
    train.update(dataroot_gt=str(dataset_root), image_manifest=str(split_dir / 'train.txt'),
                 exclude_manifest=str(holdout))
    train.pop('include_folders', None)  # The verified split is the domain authority.
    config['datasets']['val'] = {
        'name': 'FixedPortraitValidation', 'type': 'RestorationValidationDataset',
        'dataroot_gt': str(dataset_root), 'image_manifest': str(split_dir / 'validation.txt'),
        'exclude_manifest': str(holdout), 'phase': 'val', 'num_worker_per_gpu': 0}
    config['val'] = {'val_freq': 500, 'save_img': False, 'metrics': {
        'psnr': {'type': 'calculate_psnr', 'crop_border': 0, 'test_y_channel': False},
        'ssim': {'type': 'calculate_ssim', 'crop_border': 0, 'test_y_channel': False}}}
    return config, {'train_count': len(groups['train']), 'validation_count': len(groups['validation']),
                    'benchmark_unique_count': len(heldout), 'exact_pixel_overlap': 0,
                    'teacher': str(teacher), 'review_status': split.get('review_status', 'pending'),
                    'lifecycle': split.get('lifecycle'), 'review_receipt': split.get('review_receipt'),
                    'benchmark_manifest_sha256': split.get('benchmark_manifest_sha256')}
