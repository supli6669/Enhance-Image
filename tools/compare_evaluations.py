"""Compare like-for-like reports; never promote a model from smoke metrics."""
import argparse
import json
from pathlib import Path


def compare_reports(baseline, candidate):
    b, c = baseline['summary'], candidate['summary']
    if b.get('sample_set_sha256') != c.get('sample_set_sha256') or not b.get('sample_set_sha256'):
        raise ValueError('Reports must evaluate exactly the same image bytes and sample IDs')
    for key in ('fidelity', 'upscale', 'face_restore', 'postprocessing', 'identity_alignment'):
        if b['evaluation'].get(key) != c['evaluation'].get(key):
            raise ValueError(f'Evaluation settings differ: {key}')
    deltas, reasons = {}, []
    if any(s['evaluation'].get('data_review') == 'experimental' for s in (b, c)):
        reasons.append('experimental data split; production promotion not eligible')
    for key, direction in (('median_psnr', 1), ('median_ssim', 1),
                           ('median_lpips', -1), ('median_identity_similarity', 1)):
        before, after = b['overall'].get(key), c['overall'].get(key)
        if before is None or after is None:
            reasons.append(f'missing {key}')
        else:
            deltas[key] = after - before
            if (after - before) * direction < 0:
                reasons.append(f'regression in {key}')
    if b['overall'].get('identity_sample_count') != c['overall'].get('identity_sample_count'):
        reasons.append('identity metric coverage differs')
    if any(s['evaluation'].get('scope') != 'full' for s in (b, c)):
        reasons.append('smoke evaluation only')
    return {'decision': 'not_eligible' if reasons else 'human_identity_review_required',
            'reasons': reasons, 'candidate_minus_baseline': deltas,
            'baseline': b['provenance'], 'candidate': c['provenance'],
            'latency_ms': {name: s['overall'].get('median_latency_ms')
                           for name, s in [('baseline', b), ('candidate', c)]}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare_reports(json.loads(args.baseline.read_text(encoding='utf-8')),
                             json.loads(args.candidate.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(result['decision'] + ': ' + ', '.join(result['reasons']))
