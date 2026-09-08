"""Offline regression tests for review evidence and release gates."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.review_dataset import FIELDS, candidates, digest, freeze, generate, phash, safe_path, write_csv
from tools.prepare_baseline import prepare, require_reviewed_split


class ReviewTests(unittest.TestCase):
    def test_near_duplicates_and_identity_are_suggestions(self):
        image = np.random.default_rng(4).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        h = phash(image)
        entries = [{'path': 'a.png', 'split': 'train'}, {'path': 'b.png', 'split': 'holdout'}]
        rows = candidates(entries, [h, h], [np.array([1., 0.]), np.array([1., 0.])], ['', ''])
        self.assertEqual({r['reason'] for r in rows}, {'individual_quality_and_identity_review', 'near_duplicate', 'possible_same_person'})
        self.assertTrue(all(r['decision'] == 'pending' for r in rows))

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                safe_path(Path(d), '../outside.png')

    def fixture(self, directory):
        root = Path(directory)
        source, review = root / 'source', root / 'review'
        source.mkdir()
        review.mkdir()
        entries = [{'path': f't{i}.png', 'pixels_sha256': str(i), 'split': 'train'} for i in range(1002)]
        entries += [{'path': f'v{i}.png', 'pixels_sha256': f'v{i}', 'split': 'validation'} for i in range(20)]
        entries += [{'path': 'h.png', 'pixels_sha256': 'h', 'split': 'holdout'}]
        split = {group: [{k: v for k, v in r.items() if k != 'split'} for r in entries if r['split'] == group] for group in ('train', 'validation')}
        (source / 'split.json').write_text(json.dumps(split))
        (source / 'holdout_paths.txt').write_text('h.png\n')
        # Three related identities spanning all splits: holdout wins transitively.
        rows = [dict(zip(FIELDS, [str(i), a, b, 'train', 'holdout', 'possible_same_person', '.8', 'same_group', 'reviewer']))
                for i, (a, b) in enumerate([('t0.png', 't1.png'), ('t1.png', 'h.png')])]
        originals = [dict(r, decision='pending', reviewer='') for r in rows]
        write_csv(review / 'review_candidates.csv', originals)
        write_csv(review / 'decisions.csv', rows)
        record = {'identity_scanned': True, 'split_sha256': digest(source / 'split.json'),
                  'holdout_sha256': digest(source / 'holdout_paths.txt'), 'entries': entries,
                  'candidates_sha256': digest(review / 'review_candidates.csv')}
        (review / 'review.json').write_text(json.dumps(record))
        return root, source, review, split, entries, rows

    def test_transitive_group_exclusion_preserves_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            root, source, review, split, entries, _ = self.fixture(d)
            with patch('tools.review_dataset.collect', return_value=(split, entries)):
                result = freeze(root, source, review, root / 'frozen', 'human')
            self.assertEqual(len(result['train']), 1000)
            self.assertEqual(result['review_receipt']['excluded_paths'], ['t0.png', 't1.png'])
            require_reviewed_split(result)
            self.assertEqual((root / 'frozen/holdout_paths.txt').read_bytes(), (source / 'holdout_paths.txt').read_bytes())

    def test_pending_decisions_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            root, source, review, split, entries, rows = self.fixture(d)
            rows[0]['decision'] = 'pending'
            with (review / 'decisions.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with patch('tools.review_dataset.collect', return_value=(split, entries)), self.assertRaisesRegex(ValueError, 'Unresolved'):
                freeze(root, source, review, root / 'frozen', 'human')
            self.assertFalse((root / 'frozen').exists())

    def test_changed_evidence_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            root, source, review, *_ = self.fixture(d)
            with (review / 'review_candidates.csv').open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(ValueError, 'candidates changed'):
                freeze(root, source, review, root / 'frozen', 'human')

    def test_approved_string_does_not_bypass_baseline_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'split.json').write_text(json.dumps({'review_status': 'approved'}))
            with self.assertRaisesRegex(ValueError, 'freeze an explicitly reviewed'):
                prepare(root, root, root / 'missing.csv', root / 'missing.pth', root / 'run')
            self.assertFalse((root / 'run').exists())

    def test_holdout_cannot_be_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root, source, review, split, entries, rows = self.fixture(d)
            rows[1]['decision'] = 'exclude_right'
            with (review / 'decisions.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with patch('tools.review_dataset.collect', return_value=(split, entries)), self.assertRaisesRegex(ValueError, 'Holdout exclusions'):
                freeze(root, source, review, root / 'frozen', 'human')
            self.assertFalse((root / 'frozen').exists())

    def test_conflicting_groups_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            root, source, review, split, entries, rows = self.fixture(d)
            rows.append(dict(rows[0], id='2', right='h.png', decision='distinct'))
            for filename, items in [('review_candidates.csv', [dict(r, decision='pending', reviewer='') for r in rows]), ('decisions.csv', rows)]:
                with (review / filename).open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(items)
            record = json.loads((review / 'review.json').read_text())
            record['candidates_sha256'] = digest(review / 'review_candidates.csv')
            (review / 'review.json').write_text(json.dumps(record))
            with patch('tools.review_dataset.collect', return_value=(split, entries)), self.assertRaisesRegex(ValueError, 'Conflicting'):
                freeze(root, source, review, root / 'frozen', 'human')

    def test_generate_preview_creates_escaped_review_pages(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'split'
            source.mkdir()
            (source / 'split.json').write_text('{}')
            (source / 'holdout_paths.txt').write_text('a.png')
            cv2.imwrite(str(root / 'a.png'), np.zeros((24, 24, 3), dtype=np.uint8))
            entries = [{'path': 'a.png', 'pixels_sha256': 'test', 'split': 'holdout'}]
            with patch('tools.review_dataset.collect', return_value=({}, entries)):
                report = generate(root, source, root / 'review', identity=False)
            self.assertEqual(report['status'], 'pending_review')
            self.assertFalse(report['identity_scanned'])
            self.assertTrue((root / 'review/index.html').is_file())
            with self.assertRaises(FileExistsError):
                generate(root, source, root / 'review', identity=False)

    def test_interrupted_scan_resumes_completed_features(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'split'
            source.mkdir()
            (source / 'split.json').write_text('{}')
            (source / 'holdout_paths.txt').write_text('a.png\nb.png')
            entries = [{'path': name, 'pixels_sha256': name, 'split': 'holdout'} for name in ('a.png', 'b.png')]
            image = np.zeros((24, 24, 3), dtype=np.uint8)
            with patch('tools.review_dataset.collect', return_value=({}, entries)):
                with patch('tools.review_dataset.cv2.imread', side_effect=[image, RuntimeError('interrupted')]), self.assertRaises(RuntimeError):
                    generate(root, source, root / 'review', identity=False)
                self.assertTrue((root / 'review/features/000000.json').is_file())
                with patch('tools.review_dataset.cv2.imread', return_value=image) as read:
                    generate(root, source, root / 'review', identity=False, resume=True)
                self.assertEqual(read.call_count, 1)
                with self.assertRaises(FileExistsError):
                    generate(root, source, root / 'review', identity=False, resume=True)

    def test_local_ui_saves_one_explicit_decision(self):
        import os
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'split'
            source.mkdir()
            (source / 'split.json').write_text('{}')
            (source / 'holdout_paths.txt').write_text('a.png')
            cv2.imwrite(str(root / 'a.png'), np.zeros((24, 24, 3), dtype=np.uint8))
            entries = [{'path': 'a.png', 'pixels_sha256': 'test', 'split': 'holdout'}]
            with patch('tools.review_dataset.collect', return_value=({}, entries)):
                generate(root, source, root / 'review', identity=False)
            with patch.dict(os.environ, {'DATASET_REVIEW_ROOT': str(root)}):
                app = AppTest.from_file(str(Path(__file__).with_name('dataset_review_app.py'))).run(timeout=20)
                self.assertFalse(app.exception)
                app.text_input[0].input('Test reviewer')
                app.selectbox[-1].select('keep')
                app.button[0].click().run(timeout=20)
                self.assertFalse(app.exception)
            with (root / 'review/decisions.csv').open() as handle:
                decision = next(csv.DictReader(handle))
            self.assertEqual((decision['decision'], decision['reviewer']), ('keep', 'Test reviewer'))
            self.assertEqual(json.loads((root / 'review/review.json').read_text())['status'], 'pending_review')


if __name__ == '__main__':
    unittest.main()
