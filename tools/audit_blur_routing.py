"""Leave-one-source-out diagnostic of existing edge-width features; no inference router."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(samples):
    sources = sorted({r['path'] for r in samples})
    predictions = []
    folds = []
    for source in sources:
        train = [r for r in samples if r['path'] != source]
        # Reject any threshold that triggers a clean or noise/JPEG training control.
        controls = [r['profile']['edge_sigma_median'] for r in train
                    if r['condition'] in ('clean', 'noise_jpeg')
                    and r['profile']['edge_sigma_median'] is not None]
        if not controls:
            raise ValueError('No usable training controls')
        threshold = max(controls)
        folds.append({'held_out_source': source, 'threshold': threshold,
                      'training_sources': len({r['path'] for r in train})})
        for row in samples:
            if row['path'] != source:
                continue
            width = row['profile']['edge_sigma_median']
            predictions.append({'path': source, 'condition': row['condition'],
                                'available': width is not None,
                                'trigger': width is not None and width > threshold})
    groups = {}
    for condition in sorted({r['condition'] for r in predictions}):
        rows = [r for r in predictions if r['condition'] == condition]
        groups[condition] = {'count': len(rows), 'available': sum(r['available'] for r in rows),
                             'triggered': sum(r['trigger'] for r in rows),
                             'trigger_rate': sum(r['trigger'] for r in rows)/len(rows)}
    return {'scope': 'Development cross-validation only; selected feature already observed. Not independent validation.',
            'rule': 'width strictly above maximum clean/noise training width; missing width abstains',
            'folds': folds, 'groups': groups, 'predictions': predictions}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report['status'] != 'completed':
        raise ValueError('Requires a completed report')
    result = audit(report['samples'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result['groups'], indent=2))
