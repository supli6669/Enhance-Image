"""Matched/mismatched-kernel and geometry controls for experimental deblurring."""
from pathlib import Path
import sys
import unittest
import json
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.bounded_deblur import characterize, deblur


class DeblurTests(unittest.TestCase):
    def test_exact_match_metric_is_json_safe_and_explicit(self):
        from tools.evaluate_bounded_deblur import report_metrics
        metrics = report_metrics({'psnr': float('inf'), 'lpips': 0.0})
        self.assertIsNone(metrics['psnr'])
        self.assertTrue(metrics['psnr_exact_match'])
        self.assertEqual(json.loads(json.dumps(metrics, allow_nan=False)), metrics)

    def setUp(self):
        self.edge = np.full((96, 128, 3), (40, 60, 80), np.uint8)
        self.edge[:, 64:] = (160, 180, 200)

    def test_matched_blur_improves_without_moving_edge(self):
        image = cv2.GaussianBlur(self.edge, (0, 0), .9)
        out = deblur(image, sigma=.9)
        self.assertGreater(cv2.PSNR(self.edge, out), cv2.PSNR(self.edge, image))
        self.assertEqual(np.argmax(np.diff(image[48, :, 1].astype(float))), np.argmax(np.diff(out[48, :, 1].astype(float))))
        self.assertGreater(characterize(image)['valid_edges'], 0)
        self.assertGreater(characterize(image)['edge_sigma_median'], characterize(self.edge)['edge_sigma_median'])

    def test_wrong_kernels_remain_bounded_not_certified_correct(self):
        image = cv2.GaussianBlur(self.edge, (0, 0), .5)
        original = image.copy()
        for sigma in (.3, 1.5):
            out = deblur(image, sigma=sigma, upscale=2)
            base = cv2.resize(image, (256, 192), interpolation=cv2.INTER_LANCZOS4)
            delta = out.astype(int)-base.astype(int)
            np.testing.assert_array_equal(delta[:, :, 0], delta[:, :, 1])
            np.testing.assert_array_equal(delta[:, :, 1], delta[:, :, 2])
            self.assertLessEqual(np.abs(delta).max(), 16)
            for c in range(3):
                self.assertGreaterEqual(out[:, :, c].min(), base[:, :, c].min())
                self.assertLessEqual(out[:, :, c].max(), base[:, :, c].max())
        np.testing.assert_array_equal(image, original)

    def test_clean_step_flat_and_disabled_preserved(self):
        np.testing.assert_array_equal(deblur(self.edge), self.edge)
        for color in ((0, 0, 255), (120, 160, 200)):
            flat = np.full_like(self.edge, color)
            np.testing.assert_array_equal(deblur(flat), flat)
            self.assertIsNone(characterize(flat)['edge_sigma_median'])
        np.testing.assert_array_equal(deblur(self.edge, strength=0), self.edge)

    def test_invalid_controls_rejected(self):
        for args in ({'sigma': np.nan}, {'regularization': 0}, {'strength': 2}, {'upscale': 1.5}):
            with self.assertRaises(ValueError):
                deblur(self.edge, **args)

    def test_boundary_impulse_does_not_wrap_to_opposite_edge(self):
        image = np.full((65, 99, 3), 80, np.uint8)
        image[25:40, :2] = 190
        for sigma in (.6, .9, 1.5):
            out = deblur(image, sigma=sigma, strength=1)
            np.testing.assert_array_equal(out[:, -16:], image[:, -16:])
            self.assertEqual(out.shape, image.shape)


if __name__ == '__main__':
    unittest.main()
