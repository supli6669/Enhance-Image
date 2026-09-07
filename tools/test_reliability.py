"""Regression checks for the failures found in the September audit."""
import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from pipeline import LocalAIEnhancerPipeline
from tools.model_artifacts import model_files
from basicsr.data.data_util import training_paths
from basicsr.utils.training_guards import teacher_options


class ReliabilityTests(unittest.TestCase):
    def test_face_mask_keeps_semantic_classes(self):
        logits = torch.zeros(1, 19, 16, 16)
        logits[:, 1] = 1
        logits[:, 4, 2:6, 2:6] = 3
        labels = LocalAIEnhancerPipeline._parse_labels(logits)
        self.assertEqual(labels.shape, (16, 16))
        self.assertEqual(set(np.unique(labels)), {1, 4})
        self.assertEqual(labels[3, 3], 4)

    def test_cached_scale_and_lanczos_without_faces(self):
        pipe = LocalAIEnhancerPipeline.__new__(LocalAIEnhancerPipeline)
        helper = SimpleNamespace(upscale_factor=2, face_detector=SimpleNamespace(),
            clean_all=Mock(), read_image=Mock(), get_face_landmarks_5=Mock(return_value=0))
        pipe._face_helper_cache = {'retinaface_mobile0.25': helper}
        pipe.use_re_onnx = True
        pipe.enhance_realesrgan_onnx = Mock(side_effect=AssertionError('Lanczos must not call ESRGAN'))
        pipe._report_progress = Mock()
        image = np.zeros((12, 15, 3), dtype=np.uint8)
        for scale in (2, 4, 1):
            out = pipe._process_image(image, upscale=scale, bg_upsampler=None)
            self.assertEqual(out.shape, (12 * scale, 15 * scale, 3))
            self.assertEqual(helper.upscale_factor, scale)
        pipe.enhance_realesrgan_onnx.assert_not_called()

    def test_small_image_coordinates_are_preserved(self):
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        helper = FaceRestoreHelper.__new__(FaceRestoreHelper)
        image = np.zeros((123, 201, 3), dtype=np.uint8)
        helper.read_image(image, preserve_size=True)
        self.assertEqual(helper.input_img.shape, image.shape)

    def test_explicit_pytorch_selection_after_onnx(self):
        with tempfile.TemporaryDirectory() as tmp:
            pth, onnx = Path(tmp) / 'baseline.pth', Path(tmp) / 'candidate.onnx'
            pth.write_bytes(b'x' * 128); onnx.write_bytes(b'x' * 128)
            pipe = LocalAIEnhancerPipeline.__new__(LocalAIEnhancerPipeline)
            pipe.default_model_path = str(onnx)
            pipe.get_available_models = Mock(return_value={'Baseline': str(pth)})
            pipe._get_torch_model = Mock(return_value='torch-net')
            pipe._get_onnx_session = Mock(return_value='onnx-session')
            self.assertEqual(pipe._resolve_model('Auto'), ('onnx-session', None))
            self.assertEqual(pipe._resolve_model('Baseline'), (None, 'torch-net'))
            self.assertEqual(pipe.active_model_path, str(pth.resolve()))

    def test_cross_model_sidecar_rejected(self):
        import onnx
        from onnx import helper, numpy_helper
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = helper.make_graph([], 'external-test', [], [],
                [numpy_helper.from_array(np.ones(128, dtype=np.float32), name='weight')])
            model = helper.make_model(graph)
            onnx.save_model(model, str(root / 'v2.onnx'), save_as_external_data=True,
                            all_tensors_to_one_file=True, location='v2.onnx.data', size_threshold=0)
            self.assertEqual(len(model_files(root / 'v2.onnx')), 2)
            (root / 'v3.onnx').write_bytes((root / 'v2.onnx').read_bytes())
            with self.assertRaisesRegex(ValueError, 'different model sidecar'):
                model_files(root / 'v3.onnx')

    def test_nested_domains_and_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['faces/nested/a.png', 'faces/held.png', 'anime/b.png']:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'image')
            (root / 'holdout.txt').write_text('faces/held.png\n')
            found = training_paths(str(root), {'include_folders': ['faces'], 'exclude_manifest': str(root / 'holdout.txt')})
            self.assertEqual([Path(p).relative_to(root).as_posix() for p in found], ['faces/nested/a.png'])

    def test_random_teacher_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'model_path is required'):
            teacher_options({'network_vqgan': {'type': 'VQAutoEncoder'}})

    def test_ui_tabs_share_snapshot(self):
        tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
        calls = {node.func.attr: node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr in ('process_batch_images', 'process_video')}
        for call in calls.values():
            self.assertTrue(any(k.arg is None and isinstance(k.value, ast.Name)
                                and k.value.id == 'shared_process_args' for k in call.keywords))
        self.assertEqual(len(calls), 2)

    def test_no_literal_kaggle_key(self):
        for filename in ('tools/kaggle_runner.py', 'tools/download_v19_weights.py'):
            tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'KAGGLE_KEY' for t in node.targets):
                    self.assertFalse(isinstance(node.value, ast.Constant) and bool(node.value.value))


if __name__ == '__main__':
    unittest.main()
