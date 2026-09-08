"""Create a conservative experiment split without asserting identity separation."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.review_dataset import digest, safe_path


def quarantine_paths(rows, groups):
    """Quarantine non-holdout endpoints, not identity labels or inferred groups."""
    excluded, count = set(), 0
    for row in rows:
        a, b = row['left'], row['right']
        if not b:
            continue
        if a not in groups or b not in groups:
            raise ValueError('Review references an unknown image')
        if row['left_split'] != groups[a] or row['right_split'] != groups[b]:
            raise ValueError('Review split metadata changed')
        if groups[a] != groups[b]:
            count += 1
            excluded.update(name for name in (a, b) if groups[name] != 'holdout')
    return excluded, count


def technical_quality(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 256 / max(h, w)
    normalized = cv2.resize(gray, (max(1, round(w*scale)), max(1, round(h*scale))))
    sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    dark, light = float(np.mean(gray <= 5)), float(np.mean(gray >= 250))
    flags = []
    if min(h, w) < 256:
        flags.append('small_image')
    if sharpness < 30:
        flags.append('possible_global_blur')
    if dark > .2 or light > .2:
        flags.append('large_clipped_area')
    return {'width': w, 'height': h, 'sharpness': round(sharpness, 3),
            'dark_fraction': round(dark, 4), 'light_fraction': round(light, 4), 'flags': flags}


def prepare(root, source, review, output):
    if output.exists():
        raise FileExistsError('Choose a new experiment split directory')
    split = json.loads((source / 'split.json').read_text(encoding='utf-8'))
    evidence = json.loads((review / 'review.json').read_text(encoding='utf-8'))
    if evidence['split_sha256'] != digest(source / 'split.json') or evidence['holdout_sha256'] != digest(source / 'holdout_paths.txt'):
        raise ValueError('Review no longer matches the source split')
    if evidence['candidates_sha256'] != digest(review / 'review_candidates.csv'):
        raise ValueError('Candidate evidence changed')
    entries = evidence['entries']
    groups = {entry['path']: entry['split'] for entry in entries}
    if len(groups) != len(entries):
        raise ValueError('Repeated paths in source evidence')
    for group in ('train', 'validation'):
        names = [row['path'] for row in split[group]]
        if names != (source / f'{group}.txt').read_text(encoding='utf-8').splitlines():
            raise ValueError('Source manifest changed')
        if names != [row['path'] for row in entries if row['split'] == group]:
            raise ValueError('Source evidence changed')
    with (review / 'review_candidates.csv').open(encoding='utf-8', newline='') as handle:
        excluded, pair_count = quarantine_paths(list(csv.DictReader(handle)), groups)
    reasons = {name: 'cross_split_candidate_quarantine' for name in excluded}
    quality, pixels = [], {}
    for index, entry in enumerate(entries):
        name = entry['path']
        image = cv2.imread(str(safe_path(root, name)))
        if image is None:
            if groups[name] == 'holdout':
                raise ValueError('Unreadable holdout; benchmark needs repair before proceeding')
            excluded.add(name)
            reasons[name] = 'unreadable'
            quality.append({'path': name, 'split': groups[name], 'flags': ['unreadable']})
            continue
        pixel_hash = hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
        if pixel_hash != entry['pixels_sha256']:
            raise ValueError(f'Image changed since review: {name}')
        pixels.setdefault(pixel_hash, []).append(name)
        stats = technical_quality(image)
        quality.append(dict(stats, path=name, split=groups[name]))
        if 'small_image' in stats['flags'] and groups[name] != 'holdout':
            excluded.add(name)
            reasons[name] = 'small_image'
        if (index+1) % 500 == 0:
            print(f'Technical checks: {index+1}/{len(entries)}', flush=True)
    # Preserve holdout bytes; eliminate exact-pixel overlaps and within-train duplicates.
    priority = {'holdout': 0, 'validation': 1, 'train': 2}
    for names in pixels.values():
        names = sorted((n for n in names if n not in excluded), key=lambda n: (priority[groups[n]], n))
        for name in names[1:]:
            if groups[name] != 'holdout':
                excluded.add(name)
                reasons[name] = 'exact_pixel_duplicate'
    result = copy.deepcopy(split)
    result.pop('review_receipt', None)
    for group in ('train', 'validation'):
        result[group] = [row for row in split[group] if row['path'] not in excluded]
    if len(result['train']) < 1000 or len(result['validation']) < 20:
        raise ValueError('Conservative split below experiment minimum; replenish data')
    result.update(review_status='experimental', lifecycle='frozen', identity_separation_verified=False,
        mitigation={'policy': 'quarantine_all_non_holdout_cross_split_endpoints_v1',
                    'source_split_sha256': digest(source/'split.json'),
                    'candidate_sha256': digest(review/'review_candidates.csv'),
                    'cross_split_rows': pair_count, 'excluded': reasons,
                    'technical_checks_complete': True,
                    'limitation': 'Unflagged identity overlap may remain; not approved for production promotion.'})
    kept = {row['path'] for group in ('train', 'validation') for row in result[group]}
    kept.update(name for name, group in groups.items() if group == 'holdout')
    sample = []
    for group, limit in [('train', 24), ('validation', 8), ('holdout', 8)]:
        rows = [r for r in quality if r['split'] == group and r['path'] in kept]
        rows.sort(key=lambda r: hashlib.sha256(('42:'+r['path']).encode()).hexdigest())
        sample.extend(rows[:limit])
    flagged = [r for r in quality if r['flags'] and r['path'] in kept]
    # Short inspection list, separate from automatically quarantined paths.
    shortlist = flagged[:12]
    report = {'status': 'ready_for_experiment', 'train_count': len(result['train']),
        'validation_count': len(result['validation']), 'holdout_path_count': sum(g=='holdout' for g in groups.values()),
        'quarantined_count': len(excluded), 'quarantine_reasons': dict(Counter(reasons.values())),
        'quality_flag_count': len(flagged), 'sample': sample, 'shortlist': shortlist,
        'quality': quality, 'manual_decisions_required': 0,
        'identity_separation_verified': False}
    output.mkdir(parents=True)
    (output/'thumbnails').mkdir()
    for row in {r['path']: r for r in sample+shortlist}.values():
        image = cv2.imread(str(safe_path(root, row['path'])))
        h,w = image.shape[:2]
        thumb = cv2.resize(image, (max(1,round(w*192/max(h,w))),max(1,round(h*192/max(h,w)))))
        name = hashlib.sha256(row['path'].encode()).hexdigest()+'.jpg'
        if not cv2.imwrite(str(output/'thumbnails'/name), thumb):
            raise OSError('Thumbnail write failed')
    for group in ('train', 'validation'):
        (output/f'{group}.txt').write_text(''.join(r['path']+'\n' for r in result[group]), encoding='utf-8')
    (output/'holdout_paths.txt').write_bytes((source/'holdout_paths.txt').read_bytes())
    (output/'quality.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    result['mitigation']['quality_sha256'] = digest(output/'quality.json')
    (output/'split.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sample','shortlist','quality')}), flush=True)
    return result, report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'models/CodeFormer/datasets/ffhq/ffhq_512')
    p.add_argument('--source', type=Path, default=ROOT/'benchmarks/splits/real_portraits_v1')
    p.add_argument('--review', type=Path, default=ROOT/'benchmarks/reports/dataset_review_v2')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    prepare(args.root, args.source, args.review, args.output)
