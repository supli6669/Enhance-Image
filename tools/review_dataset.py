"""Generate local review evidence and freeze explicitly reviewed portrait splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_training_split import pixel_digest

FIELDS = ['id', 'left', 'right', 'left_split', 'right_split', 'reason',
          'score', 'decision', 'reviewer']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Dataset path escapes root')
    return path


def phash(image):
    gray = cv2.cvtColor(cv2.resize(image, (32, 32)), cv2.COLOR_BGR2GRAY)
    values = cv2.dct(gray.astype(np.float32))[:8, :8].flatten()[1:]
    return values > np.median(values)


class IdentityFeatures:
    """Suggestions only; ambiguous/no-face inputs always require individual review."""
    def __init__(self):
        from tools.evaluate_restoration import BenchmarkEvaluator
        import torch
        torch.set_num_threads(4)
        self.evaluator = BenchmarkEvaluator(enable_lpips=False)
        from facelib.detection import init_detection_model
        self.detector = init_detection_model('retinaface_mobile0.25', device='cpu')

    def __call__(self, image):
        import torch
        from facelib.detection.align_trans import warp_and_crop_face, get_reference_facial_points
        with torch.no_grad():
            boxes = self.detector.detect_faces(image, conf_threshold=0.8)
            count = 0 if boxes is None else len(boxes)
            if count != 1:
                return None, 'no_face' if count == 0 else 'multiple_faces'
            if min(boxes[0, 2:4] - boxes[0, :2]) < 64:
                return None, 'small_face'
            crop = warp_and_crop_face(image, boxes[0, 5:15].reshape(5, 2),
                                      reference_pts=get_reference_facial_points(default_square=True),
                                      crop_size=(112, 112))
            tensor = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                      .transpose(2, 0, 1).copy()).float()[None] / 127.5 - 1
            vector = self.evaluator.arcface_fn(tensor).flatten().numpy()
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm <= 0:
            raise ValueError('Invalid identity embedding')
        return vector / norm, ''


def collect(root, split_dir):
    split = json.loads((split_dir / 'split.json').read_text(encoding='utf-8'))
    entries = []
    seen = set()
    for group in ('train', 'validation'):
        manifest = (split_dir / f'{group}.txt').read_text(encoding='utf-8').splitlines()
        if manifest != [r['path'] for r in split[group]]:
            raise ValueError('Split manifest was modified')
        for row in split[group]:
            if row['path'] in seen:
                raise ValueError('Path occurs in more than one split')
            seen.add(row['path'])
            path = safe_path(root, row['path'])
            if pixel_digest(path) != row['pixels_sha256']:
                raise ValueError('Image changed after split preparation')
            entries.append(dict(row, split=group))
    for name in sorted(set((split_dir / 'holdout_paths.txt').read_text(encoding='utf-8').splitlines())):
        if not name:
            continue
        if name in seen:
            raise ValueError('Holdout path overlaps training/validation')
        seen.add(name)
        entries.append({'path': name, 'pixels_sha256': pixel_digest(safe_path(root, name)), 'split': 'holdout'})
    return split, entries


def candidates(entries, hashes, vectors, issues, hamming=6, cosine=.55):
    rows = []
    def add(i, j, reason, score=''):
        a, b = entries[i], entries[j] if j is not None else {}
        rows.append(dict(zip(FIELDS, [f'R{len(rows)+1:06d}', a['path'], b.get('path', ''),
            a['split'], b.get('split', ''), reason, str(score), 'pending', ''])))
    matrix = np.asarray(hashes, dtype=bool)
    for i, entry in enumerate(entries):
        # Individual review covers defects and same-person matches missed by thresholds.
        add(i, None, issues[i] or 'individual_quality_and_identity_review')
        distances = np.count_nonzero(matrix[i+1:] != matrix[i], axis=1)
        for offset in np.flatnonzero(distances <= hamming):
            j = i + 1 + int(offset)
            add(i, j, 'near_duplicate', int(distances[offset]))
    indices = [i for i, v in enumerate(vectors) if v is not None]
    if indices:
        features = np.asarray([vectors[i] for i in indices], dtype=np.float32)
        # Bounded memory; never allocate an N x N similarity matrix.
        for start in range(0, len(indices), 128):
            similarities = features[start:start+128] @ features.T
            for local, scores in enumerate(similarities):
                a = start + local
                for b in np.flatnonzero(scores[a+1:] >= cosine) + a + 1:
                    add(indices[a], indices[int(b)], 'possible_same_person', round(float(scores[b]), 5))
    return rows


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def generate(root, split_dir, output, identity=True, hamming=6, cosine=.55, resume=False):
    if output.exists() and not resume:
        raise FileExistsError('Choose a new review directory; evidence is never overwritten')
    split, entries = collect(root, split_dir)
    signature = {'entries': entries, 'identity': identity, 'hamming': hamming, 'cosine': cosine,
                 'split_sha256': digest(split_dir / 'split.json'),
                 'holdout_sha256': digest(split_dir / 'holdout_paths.txt'),
                 'code_sha256': digest(Path(__file__))}
    if identity:
        signature['arcface_sha256'] = digest(ROOT / 'weights/facelib/recognition_arcface_ir_se50.pth')
    if resume:
        if (output / 'review.json').exists():
            raise FileExistsError('Review already completed; do not overwrite decisions')
        if json.loads((output / 'scan.json').read_text(encoding='utf-8')) != signature:
            raise ValueError('Scan inputs, configuration or code changed; choose a new directory')
    else:
        output.mkdir(parents=True)
        (output / 'thumbnails').mkdir()
        (output / 'features').mkdir()
        (output / 'scan.json').write_text(json.dumps(signature), encoding='utf-8')
    extractor = IdentityFeatures() if identity else None
    hashes, vectors, issues, thumbs = [], [], [], {}
    for index, entry in enumerate(entries):
        name = f'thumbnails/{index:06d}.jpg'
        feature_path = output / f'features/{index:06d}.json'
        if feature_path.is_file() and (output / name).is_file():
            saved = json.loads(feature_path.read_text(encoding='utf-8'))
            hashes.append(np.asarray(saved['phash'], dtype=bool))
            vectors.append(np.asarray(saved['vector'], dtype=np.float32) if saved['vector'] is not None else None)
            issues.append(saved['issue'])
            thumbs[entry['path']] = name
            continue
        image = cv2.imread(str(safe_path(root, entry['path'])))
        hashes.append(phash(image))
        vector, issue = extractor(image) if extractor else (None, 'identity_not_scanned')
        vectors.append(vector)
        issues.append(issue)
        thumb = cv2.resize(image, (max(1, round(image.shape[1]*min(192/image.shape[1], 192/image.shape[0]))),
                                   max(1, round(image.shape[0]*min(192/image.shape[1], 192/image.shape[0])))))
        if not cv2.imwrite(str(output / name), thumb):
            raise OSError('Could not write thumbnail')
        thumbs[entry['path']] = name
        partial = feature_path.with_suffix('.partial')
        partial.write_text(json.dumps({'phash': hashes[-1].tolist(),
            'vector': vector.tolist() if vector is not None else None, 'issue': issue}), encoding='utf-8')
        partial.replace(feature_path)
        if (index+1) % 100 == 0:
            print(f'Review features: {index+1}/{len(entries)}', flush=True)
    rows = candidates(entries, hashes, vectors, issues, hamming, cosine)
    write_csv(output / 'review_candidates.csv', rows)
    write_csv(output / 'decisions.csv', rows)
    pages = []
    for start in range(0, len(rows), 60):
        name = f'page_{start//60+1:04d}.html'
        pages.append(name)
        cards = []
        for row in rows[start:start+60]:
            pictures = ''.join(f'<figure><img src="{thumbs[row[k]]}"><figcaption>{html.escape(row[k])}</figcaption></figure>'
                               for k in ('left', 'right') if row[k])
            cards.append(f'<article><h3>{row["id"]} — {row["reason"]} ({row["score"]})</h3>'
                         f'<p>{row["left_split"]} / {row["right_split"]}</p>{pictures}</article>')
        (output / name).write_text('<meta charset="utf-8"><style>body{font:16px sans-serif}article{border-bottom:1px solid #aaa}figure{display:inline-block;max-width:280px;vertical-align:top}figcaption{overflow-wrap:anywhere}</style><a href="index.html">Index</a>'+''.join(cards), encoding='utf-8')
    links = ''.join(f'<li><a href="{name}">{name}</a></li>' for name in pages)
    (output / 'index.html').write_text('<meta charset="utf-8"><h1>Private dataset review</h1><p>Suggestions are not identity labels. Review every image and pair. Edit decisions.csv: individual keep/exclude; pairs distinct/same_group/exclude_left/exclude_right/exclude_both. Fill reviewer. Holdout images cannot be excluded here. Unresolved items remain pending.</p><ul>'+links+'</ul>', encoding='utf-8')
    record = {'schema_version': 1, 'status': 'pending_review', 'identity_scanned': identity,
              'thresholds': {'phash_hamming': hamming, 'identity_cosine': cosine},
              'source_split': split, 'entries': entries, 'candidate_count': len(rows),
              'candidates_sha256': digest(output / 'review_candidates.csv'),
              'split_sha256': digest(split_dir / 'split.json'),
              'holdout_sha256': digest(split_dir / 'holdout_paths.txt')}
    record['scan_sha256'] = digest(output / 'scan.json')
    (output / 'review.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Review ready: {len(entries)} images, {len(rows)} items, {len(pages)} pages; pending human review', flush=True)
    return record


def freeze(root, split_dir, review_dir, output, reviewer):
    if output.exists():
        raise FileExistsError('Choose a new frozen split directory')
    record = json.loads((review_dir / 'review.json').read_text(encoding='utf-8'))
    if not reviewer.strip() or not record['identity_scanned']:
        raise ValueError('Named reviewer and identity scan are required')
    if digest(split_dir / 'split.json') != record['split_sha256'] or digest(split_dir / 'holdout_paths.txt') != record['holdout_sha256']:
        raise ValueError('Source split changed during review')
    if digest(review_dir / 'review_candidates.csv') != record['candidates_sha256']:
        raise ValueError('Review candidates changed; edit only decisions.csv')
    split, entries = collect(root, split_dir)
    if entries != record['entries']:
        raise ValueError('Dataset changed during review')
    def read(name):
        with (review_dir / name).open(encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))
    originals, decisions = read('review_candidates.csv'), read('decisions.csv')
    if len(originals) != len(decisions):
        raise ValueError('Missing review decisions')
    groups = {r['path']: r['split'] for r in entries}
    parent = {name: name for name in groups}
    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name
    excluded, distinct = set(), []
    for original, row in zip(originals, decisions):
        if any(original[k] != row[k] for k in FIELDS if k not in ('decision', 'reviewer')):
            raise ValueError('Review row metadata changed')
        if not row['reviewer'].strip():
            raise ValueError('Every review item requires a reviewer')
        a, b, decision = row['left'], row['right'], row['decision']
        allowed = {'distinct', 'same_group', 'exclude_left', 'exclude_right', 'exclude_both'} if b else {'keep', 'exclude'}
        if decision not in allowed:
            raise ValueError(f'Unresolved or invalid decision: {row["id"]}')
        if decision in ('exclude', 'exclude_left', 'exclude_both'):
            excluded.add(a)
        if decision in ('exclude_right', 'exclude_both'):
            excluded.add(b)
        if decision == 'same_group':
            parent[find(a)] = find(b)
        if decision == 'distinct':
            distinct.append((a, b))
    if any(find(a) == find(b) for a, b in distinct):
        raise ValueError('Conflicting identity group decisions')
    components = {}
    for name in groups:
        components.setdefault(find(name), []).append(name)
    priority = {'train': 0, 'validation': 1, 'holdout': 2}
    for names in components.values():
        highest = max(priority[groups[name]] for name in names)
        excluded.update(name for name in names if priority[groups[name]] < highest)
    if any(groups[name] == 'holdout' for name in excluded):
        raise ValueError('Holdout exclusions require a new benchmark version and baseline')
    for group in ('train', 'validation'):
        split[group] = [row for row in split[group] if row['path'] not in excluded]
    pixels = {}
    for row in entries:
        if row['path'] in excluded:
            continue
        previous = pixels.setdefault(row['pixels_sha256'], row['split'])
        if previous != row['split']:
            raise ValueError('Exact pixel leakage remains after review')
    if len(split['train']) < 1000 or len(split['validation']) < 20:
        raise ValueError('Reviewed split is below production minimum; replenish data first')
    receipt = {'reviewer': reviewer, 'review_sha256': digest(review_dir / 'review.json'),
               'decisions_sha256': digest(review_dir / 'decisions.csv'), 'excluded_paths': sorted(excluded),
               'identity_scan': True, 'all_items_reviewed': True}
    split.update(review_status='approved', lifecycle='frozen', review_receipt=receipt)
    output.mkdir(parents=True)
    for group in ('train', 'validation'):
        (output / f'{group}.txt').write_text(''.join(row['path']+'\n' for row in split[group]), encoding='utf-8')
    (output / 'holdout_paths.txt').write_bytes((split_dir / 'holdout_paths.txt').read_bytes())
    (output / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    print(f'Frozen reviewed split: {len(split["train"])} train, {len(split["validation"])} validation')
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['generate', 'freeze'])
    parser.add_argument('--root', type=Path, default=ROOT / 'models/CodeFormer/datasets/ffhq/ffhq_512')
    parser.add_argument('--split-dir', type=Path, default=ROOT / 'benchmarks/splits/real_portraits_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-identity', action='store_true', help='Preview only; cannot freeze')
    parser.add_argument('--resume', action='store_true', help='Resume an interrupted feature scan with identical inputs')
    parser.add_argument('--review-dir', type=Path)
    parser.add_argument('--reviewer', default='')
    args = parser.parse_args()
    if args.action == 'generate':
        generate(args.root, args.split_dir, args.output, identity=not args.skip_identity, resume=args.resume)
    else:
        if args.review_dir is None:
            parser.error('--review-dir required for freeze')
        freeze(args.root, args.split_dir, args.review_dir, args.output, args.reviewer)


if __name__ == '__main__':
    main()
