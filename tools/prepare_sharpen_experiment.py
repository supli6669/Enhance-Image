"""Freeze development/check partitions for regional sharpening; never overwrite."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset = ROOT / 'models/CodeFormer/datasets/ffhq/ffhq_512'
    split = ROOT / 'benchmarks/splits/portraits_experiment_v1/validation.txt'
    names = split.read_text().splitlines()
    observed = {names[i] for n in (3, 6) for i in np.linspace(0, len(names)-1, n, dtype=int)}
    previous = ROOT / 'artifacts/enhance_quality_check/adaptive_sharpen_final/report.json'
    observed.update(row['path'] for row in json.loads(previous.read_text())['samples'])
    entries, hashes = [], []
    for name in names:
        path = (dataset / name).resolve()
        if not path.is_relative_to(dataset.resolve()) or not path.is_file():
            raise ValueError(f'Unsafe or missing path: {name}')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable: {name}')
        small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
        block = cv2.dct(small)[:8, :8].flatten()[1:]
        bits = block > np.median(block)
        hashes.append(bits)
        entries.append({'path': name, 'sha256': digest(path),
                        'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
    # Group image-level similarity suggestions, not facial identities.
    parents = list(range(len(names)))
    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i
    suggestions = []
    for i in range(len(names)):
        for j in range(i):
            distance = int(np.count_nonzero(hashes[i] != hashes[j]))
            if distance <= 8 or entries[i]['pixel_sha256'] == entries[j]['pixel_sha256']:
                parents[root(i)] = root(j)
                suggestions.append({'left': names[i], 'right': names[j], 'phash_distance': distance})
    excluded_groups = {root(i) for i, name in enumerate(names) if name in observed}
    groups = {}
    for i in range(len(names)):
        if root(i) not in excluded_groups:
            groups.setdefault(root(i), []).append(i)
    order = np.random.default_rng(9132026).permutation(list(groups))
    partitions = {'development': [], 'check': [], 'reserve': []}
    for group in order:
        target = next((key for key in ('development', 'check') if len(partitions[key]) < 24), 'reserve')
        partitions[target].extend(entries[i] for i in groups[group])
    if any(len(partitions[k]) < 24 for k in ('development', 'check')):
        raise ValueError('Insufficient independent image groups for planned split')
    report = {'scope': 'experimental validation; no identity separation claim', 'seed': 9132026,
              'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'validation_sha256': digest(split), 'previous_check_sha256': digest(previous),
              'plan_sha256': digest(ROOT/'DETAIL_PRESERVING_SHARPEN_PLAN.md'),
              'observed_excluded': sorted(observed), 'similarity_suggestions': suggestions,
              'partitions': partitions, 'coverage_limitations': ['Portrait-only real data; no confirmed coverage labels for glasses, pose or small faces.', 'General-image properties use synthetic regression fixtures.'],
              'variants': ['baseline', 'regional_noise', 'regional_structure', 'guided_structure'],
              'strengths': [0.35, 0.50, 0.65],
              'conditions': ['clean', 'soft', 'severe', 'noise_jpeg', 'motion_h', 'motion_diagonal', 'anisotropic'],
              'check_policy': 'Do not evaluate check until a development candidate passes and config is frozen. Never adjust thresholds after viewing it.'}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name in ('wink_enhancer.py', 'pipeline.py', 'DETAIL_PRESERVING_SHARPEN_PLAN.md'):
        (args.output/name).write_bytes((ROOT/name).read_bytes())
    print(json.dumps({'counts': {k: len(v) for k, v in partitions.items()}, 'excluded': len(observed), 'suggestions': len(suggestions)}))


if __name__ == '__main__':
    main()
