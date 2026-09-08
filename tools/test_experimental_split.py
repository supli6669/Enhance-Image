"""Offline tests for conservative quarantine, quick review and experiment gates."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_experimental_split import quarantine_paths, technical_quality
from tools.prepare_baseline import require_experimental_split, require_reviewed_split
from tools.review_dataset import digest
from tools.compare_evaluations import compare_reports


class ExperimentTests(unittest.TestCase):
    def test_quarantine_all_non_holdout_endpoints_without_identity_labels(self):
        groups = {'a':'train','b':'validation','c':'holdout','d':'train'}
        rows = [{'left':a,'right':b,'left_split':groups[a],'right_split':groups[b]}
                for a,b in [('a','b'),('b','c'),('a','d')]]
        excluded, count = quarantine_paths(rows,groups)
        self.assertEqual(excluded,{'a','b'})
        self.assertEqual(count,2)
        self.assertNotIn('c',excluded)

    def test_stale_split_labels_rejected(self):
        with self.assertRaises(ValueError):
            quarantine_paths([{'left':'a','right':'b','left_split':'holdout','right_split':'validation'}],{'a':'train','b':'validation'})

    def test_technical_quality_only_flags_small_flat_image(self):
        result=technical_quality(np.zeros((64,64,3),dtype=np.uint8))
        self.assertIn('small_image',result['flags'])
        self.assertIn('possible_global_blur',result['flags'])

    def fixture(self, directory):
        root=Path(directory)
        reports=root/'reports'
        reports.mkdir()
        experiment=root/'splits'/'experiment'
        experiment.mkdir(parents=True)
        (experiment/'thumbnails').mkdir()
        row={'path':'a.png','split':'train','flags':[]}
        cv2.imwrite(str(experiment/'thumbnails'/(hashlib.sha256(b'a.png').hexdigest()+'.jpg')),np.zeros((24,24,3),np.uint8))
        quality={'train_count':2400,'validation_count':120,'quarantined_count':60,
                 'sample':[row],'shortlist':[],'quality_flag_count':0}
        (experiment/'quality.json').write_text(json.dumps(quality))
        split={'review_status':'experimental','lifecycle':'frozen','identity_separation_verified':False,
               'mitigation':{'policy':'quarantine_all_non_holdout_cross_split_endpoints_v1',
                             'technical_checks_complete':True,'quality_sha256':digest(experiment/'quality.json'),
                             'excluded':{'b.png':'cross_split_candidate_quarantine'}}}
        (experiment/'split.json').write_text(json.dumps(split))
        return reports,experiment,split

    def test_experiment_is_allowed_only_by_explicit_experiment_gate(self):
        with tempfile.TemporaryDirectory() as d:
            _,directory,split=self.fixture(d)
            require_experimental_split(split,directory)
            with self.assertRaises(ValueError):
                require_reviewed_split(split)
            (directory/'quality.json').write_text('{}')
            with self.assertRaises(ValueError):
                require_experimental_split(split,directory)

    def test_experimental_report_cannot_promote_model(self):
        report={'summary':{'sample_set_sha256':'same','evaluation':{'scope':'full','data_review':'experimental'},
                          'overall':{'median_psnr':30,'median_ssim':.8,'median_lpips':.2,'median_identity_similarity':.9},
                          'provenance':{}}}
        compared=compare_reports(report,copy.deepcopy(report))
        self.assertEqual(compared['decision'],'not_eligible')
        self.assertIn('experimental',compared['reasons'][0])

    def test_quick_ui_requires_no_manual_decisions(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as d:
            reports,_,_=self.fixture(d)
            with patch.dict(os.environ,{'DATASET_REVIEW_ROOT':str(reports)}):
                app=AppTest.from_file(str(ROOT/'tools/dataset_review_app.py')).run(timeout=20)
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value,'2400')
                self.assertFalse(app.button)
                self.assertFalse(app.text_input)
                self.assertEqual(len(app.tabs),3)


if __name__=='__main__':
    unittest.main()
