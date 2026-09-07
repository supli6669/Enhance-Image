"""Offline checks for unsafe downloads, calibration and validation contracts."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.artifact_download import destination, download
from tools.quantize_onnx_static import CodeFormerCalibrationDataReader
from tools.evaluate_restoration import compute_summary, SampleMetricResult
from basicsr.data.restoration_validation_dataset import RestorationValidationDataset
from tools.compare_evaluations import compare_reports
from basicsr.data.gaussian_kernels import motion_kernels_32


class WorkflowTests(unittest.TestCase):
    def test_training_validation_denormalizes_images(self):
        import torch
        from types import SimpleNamespace
        from basicsr.models.codeformer_joint_model import CodeFormerJointModel
        model = CodeFormerJointModel.__new__(CodeFormerJointModel)
        model.device = torch.device('cpu')
        model.opt = {'val': {'metrics': {'psnr': {'type': 'calculate_psnr'}}}}
        model.test = lambda: setattr(model, 'output', model.gt.clone())
        model._log_validation_metric_values = Mock()
        class Loader(list):
            dataset = SimpleNamespace(opt={'name': 'fixture'})
        pixels = torch.full((1, 3, 16, 16), -.5)
        loader = Loader([{'gt': pixels, 'in': pixels, 'in_large_de': pixels, 'lq_path': ['sample.png']}])
        with patch('basicsr.models.codeformer_joint_model.calculate_metric', return_value=42.) as metric:
            model.nondist_validation(loader, 1, None, False)
        self.assertEqual(int(metric.call_args.args[0]['img1'][0, 0, 0]), 64)
        self.assertEqual(model.metric_results['psnr'], 42.)

    def test_smoke_report_cannot_promote_model(self):
        from copy import deepcopy
        report = {'summary': {'sample_set_sha256': 'same', 'provenance': {},
            'evaluation': {'scope': 'smoke'}, 'overall': {'median_psnr': 30., 'median_ssim': .8,
             'median_lpips': .2, 'median_identity_similarity': .9, 'identity_sample_count': 3}}}
        result = compare_reports(report, deepcopy(report))
        self.assertEqual(result['decision'], 'not_eligible')
        self.assertIn('smoke evaluation only', result['reasons'])

    def test_motion_kernels_conserve_brightness(self):
        kernels = motion_kernels_32()
        self.assertEqual(len(kernels), 32)
        for kernel in kernels.values():
            self.assertAlmostEqual(float(kernel.sum()), 1., places=5)
            self.assertTrue((kernel >= 0).all())
        self.assertFalse(np.array_equal(kernels['00'], kernels['16']))

    def test_download_path_traversal(self):
        for name in ('../net.pth', '/tmp/net.pth', 'C:/net.pth', 'a/../../net.pth', '..\\net.pth'):
            with self.assertRaises(ValueError):
                destination(ROOT / 'artifacts', name)

    def test_download_failure_preserves_active_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'active.pth'
            path.write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                download('https://example.invalid/model', path)
            self.assertEqual(path.read_bytes(), b'original')
            path = Path(tmp) / 'new.pth'
            response = Mock(status_code=200, headers={'content-length': '10'})
            response.iter_content.return_value = iter([b'short'])
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            with patch('tools.artifact_download.requests.get', return_value=response):
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    download('https://example.invalid/model', path)
            self.assertFalse(path.exists())

    def test_empty_calibration_never_generates_random_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'real calibration'):
                CodeFormerCalibrationDataReader(tmp)

    def test_validation_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv2.imwrite(str(root / 'val.png'), np.full((64, 64, 3), 120, np.uint8))
            (root / 'validation.txt').write_text('val.png\n')
            dataset = RestorationValidationDataset({'dataroot_gt': str(root), 'image_manifest': str(root / 'validation.txt')})
            a, b = dataset[0], dataset[0]
            self.assertEqual(tuple(a['in'].shape), (3, 512, 512))
            self.assertTrue(a['in'].equal(b['in']))
            self.assertFalse(a['in'].equal(a['gt']))

    def test_summary_uses_median_and_reports_missing_identity(self):
        rows = [SampleMetricResult(str(i), 'blur', latency_ms=i, psnr=p, ssim=.8) for i,p in enumerate((10., 20., 90.))]
        stats = compute_summary(rows)['overall']
        self.assertEqual(stats['median_psnr'], 20.)
        self.assertEqual(stats['identity_sample_count'], 0)
        self.assertIsNone(stats['mean_identity_similarity'])


if __name__ == '__main__':
    unittest.main()
