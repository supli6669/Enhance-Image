"""Create reproducible real-portrait splits, excluding benchmark duplicates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2


def pixel_digest(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Unreadable dataset image: {path}')
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def prepare_split(root: Path, holdout: Path, output: Path, folders=('faces', 'pinterest'), seed=42, exclude_unreadable=False):
    root = root.resolve()
    excluded = {s.strip() for s in holdout.read_text(encoding='utf-8').splitlines() if s.strip()}
    heldout_hashes = set()
    for name in sorted(excluded):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Holdout path escapes dataset root')
        heldout_hashes.add(pixel_digest(path))
    seen, rows, rejected = set(), [], []
    duplicates = 0
    for folder in folders:
        for path in sorted((root / folder).rglob('*')):
            if not path.is_file() or path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            try:
                digest = pixel_digest(path)
            except ValueError:
                if not exclude_unreadable:
                    raise
                rejected.append({'path': path.relative_to(root).as_posix(), 'reason': 'unreadable'})
                continue
            if digest in heldout_hashes or digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            rows.append({'path': path.relative_to(root).as_posix(), 'pixels_sha256': digest})
    if len(rows) < 100:
        raise ValueError(f'Only {len(rows)} unique non-holdout portraits; refusing a toy split')
    rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['pixels_sha256']}".encode()).hexdigest())
    validation_count = max(1, round(len(rows) * .05))
    validation, train = rows[:validation_count], rows[validation_count:]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new split directory to preserve existing splits')
    output.mkdir(parents=True, exist_ok=True)
    for name, items in [('train', train), ('validation', validation)]:
        (output / f'{name}.txt').write_text(''.join(row['path'] + '\n' for row in items), encoding='utf-8')
    report = {'schema_version': 1, 'seed': seed, 'domains': list(folders),
              'holdout_count': len(excluded), 'excluded_or_duplicate_count': duplicates,
              'train': train, 'validation': validation, 'rejected': rejected,
              'review_status': 'pending_identity_and_near_duplicate_review'}
    benchmark_manifest = holdout.parent / 'manifest.csv'
    if benchmark_manifest.is_file():
        report['benchmark_manifest_sha256'] = hashlib.sha256(benchmark_manifest.read_bytes()).hexdigest()
    (output / 'holdout_paths.txt').write_text(holdout.read_text(encoding='utf-8'), encoding='utf-8')
    (output / 'split.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Split: {len(train)} train, {len(validation)} validation; {len(excluded)} holdout sources excluded')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('models/CodeFormer/datasets/ffhq/ffhq_512'))
    parser.add_argument('--holdout', type=Path, default=Path('benchmarks/holdout_paths.txt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-unreadable', action='store_true', help='Record unreadable images as excluded; never delete them')
    args = parser.parse_args()
    prepare_split(args.root, args.holdout, args.output, exclude_unreadable=args.exclude_unreadable)
