"""Behavioral checks for experimental maps; production defaults are untouched."""
import sys
from pathlib import Path
import unittest
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.regional_sharpen import analyze, sharpen


class RegionalTests(unittest.TestCase):
    def test_mixed_noise_is_estimated_locally(self):
        rng = np.random.default_rng(31)
        image = np.full((128, 256, 3), 120., np.float32)
        image[:, :128] += rng.normal(0, 1, (128, 128, 3))
        image[:, 128:] += rng.normal(0, 10, (128, 128, 3))
        maps = analyze(np.clip(image, 0, 255).astype(np.uint8))
        self.assertGreater(maps['noise_sigma'][:, 160:].mean(), maps['noise_sigma'][:, :96].mean()*4)
        for key in ('noise_confidence', 'detail_support', 'edge_protection'):
            self.assertTrue(np.isfinite(maps[key]).all())
            self.assertGreaterEqual(maps[key].min(), 0)
            self.assertLessEqual(maps[key].max(), 1)

    def test_soft_edge_preserves_color_and_has_no_new_extrema(self):
        edge = np.full((97, 99, 3), (50, 70, 90), np.uint8)
        edge[:, 50:] = (170, 190, 210)
        image = cv2.GaussianBlur(edge, (0, 0), 1.2)
        for variant in ('regional_noise', 'regional_structure', 'guided_structure'):
            for scale in (1, 2, 4):
                output = sharpen(image, .5, scale, variant)
                base = cv2.resize(image, (99*scale, 97*scale), interpolation=cv2.INTER_LANCZOS4)
                delta = output.astype(int)-base.astype(int)
                np.testing.assert_array_equal(delta[:, :, 0], delta[:, :, 1])
                np.testing.assert_array_equal(delta[:, :, 1], delta[:, :, 2])
                for c in range(3):
                    self.assertGreaterEqual(output[:, :, c].min(), base[:, :, c].min())
                    self.assertLessEqual(output[:, :, c].max(), base[:, :, c].max())

    def test_zero_strength_and_flat_colors(self):
        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, (65, 67, 3), dtype=np.uint8)
        np.testing.assert_array_equal(sharpen(image, 0), image)
        for color in ((0, 0, 255), (80, 120, 170), (255, 255, 255)):
            flat = np.full_like(image, color)
            np.testing.assert_array_equal(sharpen(flat, 1), flat)

    def test_map_shape_and_range_rejected(self):
        image = np.full((65, 67, 3), 120, np.uint8)
        maps = analyze(image)
        maps['detail_support'] = np.ones((1, 67), np.float32)
        with self.assertRaises(ValueError):
            sharpen(image, maps=maps)
        maps = analyze(image)
        maps['detail_support'][0, 0] = 2
        with self.assertRaises(ValueError):
            sharpen(image, maps=maps)

    def test_text_has_more_detail_support_than_flat_noise(self):
        image = np.full((128, 256, 3), 110, np.uint8)
        cv2.putText(image, 'TEXT', (5, 75), cv2.FONT_HERSHEY_SIMPLEX, .8, (220, 220, 220), 2)
        rng = np.random.default_rng(14)
        image[:, 160:] = np.clip(110+rng.normal(0, 8, (128, 96, 3)), 0, 255).astype(np.uint8)
        original = image.copy()
        maps = analyze(image)
        self.assertGreater(np.percentile(maps['detail_support'][40:85, 5:100], 90),
                           np.percentile(maps['detail_support'][20:100, 180:240], 90))
        sharpen(image, maps=maps)
        np.testing.assert_array_equal(image, original)


if __name__ == '__main__':
    unittest.main()
