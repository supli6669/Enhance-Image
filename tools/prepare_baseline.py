"""Prepare or execute a full baseline only after an explicitly reviewed split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.review_dataset import digest


def require_reviewed_split(split):
    receipt = split.get('review_receipt') or {}
    if (split.get('review_status') != 'approved' or split.get('lifecycle') != 'frozen'
            or not receipt.get('reviewer', '').strip()
            or receipt.get('identity_scan') is not True
            or receipt.get('all_items_reviewed') is not True
            or not receipt.get('decisions_sha256') or not receipt.get('review_sha256')):
        raise ValueError('Baseline blocked: freeze an explicitly reviewed split first')


def prepare(root, split_dir, manifest, model, output, execute=False):
    split = json.loads((split_dir / 'split.json').read_text(encoding='utf-8'))
    require_reviewed_split(split)
    if output.exists():
        raise FileExistsError('Choose a new baseline run directory')
    if digest(manifest) != split.get('benchmark_manifest_sha256'):
        raise ValueError('Benchmark changed; version the split and benchmark together')
    import yaml
    from tools.training_preflight import configure_training
    from tools.model_artifacts import describe_model
    from tools.evaluate_restoration import load_manifest, validate_files
    from tools.prepare_training_split import pixel_digest
    from tools.review_dataset import safe_path
    config = yaml.safe_load((ROOT / 'models/CodeFormer/options/CodeFormer_stage3_custom.yml').read_text(encoding='utf-8'))
    _, verified = configure_training(config, root, split_dir, split_dir / 'holdout_paths.txt')
    samples = load_manifest(manifest)
    validate_files(samples)
    references = {pixel_digest(s.reference_path) for s in samples if s.reference_path is not None}
    holdout = {pixel_digest(safe_path(root, name)) for name in (split_dir / 'holdout_paths.txt').read_text(encoding='utf-8').splitlines() if name}
    if any(s.reference_path is None for s in samples) or references != holdout:
        raise ValueError('Benchmark references do not exactly match reviewed holdout pixels')
    if len(references) < 300:
        raise ValueError('Production baseline requires at least 300 unique references')
    command = [sys.executable, '-B', '-u', str(ROOT / 'tools/evaluate_restoration.py'),
               '--manifest', str(manifest.resolve()), '--model', str(model.resolve()),
               '--output-json', str(output.resolve() / 'report.json')]
    record = {'status': 'ready', 'split_sha256': digest(split_dir / 'split.json'),
              'manifest_sha256': digest(manifest), 'dataset_verification': verified,
              'model': describe_model(model), 'command': command,
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'full', 'timing_note': 'Measure controlled CPU latency separately from quality evaluation'}
    output.mkdir(parents=True)
    path = output / 'run.json'
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    if execute:
        record['status'] = 'running'
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        with (output / 'evaluation.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record['status'] = 'completed' if result.returncode == 0 else 'failed'
        record['exit_code'] = result.returncode
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Baseline failed; inspect {output / "evaluation.log"}')
    print(f'Baseline {record["status"]}: {path}')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'models/CodeFormer/datasets/ffhq/ffhq_512')
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'benchmarks/manifest.csv')
    parser.add_argument('--model', type=Path, default=ROOT / 'weights/CodeFormer/codeformer.pth')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    prepare(args.root, args.split_dir, args.manifest, args.model, args.output, args.execute)
