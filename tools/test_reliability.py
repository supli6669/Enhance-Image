"""Regression checks for the failures found in the September audit."""
import ast
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import cv2
import torch
from pipeline import LocalAIEnhancerPipeline
from wink_enhancer import WinkQualityEnhancer
from tools.model_artifacts import model_files
from basicsr.data.data_util import training_paths
from basicsr.utils.training_guards import teacher_options


class ClarityFidelityTests(unittest.TestCase):
    def setUp(self):
        self.enhancer = WinkQualityEnhancer()
        self.edge = np.full((97, 99, 3), 60, dtype=np.uint8)
        self.edge[:, 50:] = 190

    def test_near_zero_strength_does_not_change_image(self):
        for effect in (self.enhancer.apply_laplacian_pyramid_clarity,
                       self.enhancer.apply_dehaze_and_dynamic_contrast):
            for strength in (0.0, 0.0001):
                with self.subTest(effect=effect.__name__, strength=strength):
                    np.testing.assert_array_equal(effect(self.edge, strength=strength), self.edge)

    def test_clarity_recovers_soft_edges_without_overshoot_or_color_shift(self):
        sharp = self.edge.copy()
        sharp[:, :, 0] -= 20
        sharp[:, :, 2] += 20
        blurred = cv2.GaussianBlur(sharp, (0, 0), 1.2)
        original = blurred.copy()
        out = self.enhancer.apply_laplacian_pyramid_clarity(blurred, strength=0.35)
        np.testing.assert_array_equal(blurred, original)
        self.assertEqual(out.shape, blurred.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(cv2.PSNR(sharp, out), cv2.PSNR(sharp, blurred))
        self.assertGreater(float(np.abs(np.diff(out.astype(float), axis=1)).max()),
                           float(np.abs(np.diff(blurred.astype(float), axis=1)).max()))
        delta = out.astype(int) - blurred.astype(int)
        self.assertLessEqual(np.abs(delta).max(), 6)
        np.testing.assert_array_equal(delta[:, :, 0], delta[:, :, 1])
        np.testing.assert_array_equal(delta[:, :, 1], delta[:, :, 2])
        for channel in range(3):
            self.assertGreaterEqual(out[:, :, channel].min(), sharp[:, :, channel].min())
            self.assertLessEqual(out[:, :, channel].max(), sharp[:, :, channel].max())

    def test_flat_color_and_low_noise_are_preserved(self):
        for color in ((0, 0, 255), (60, 100, 150), (255, 255, 255)):
            flat = np.full((65, 67, 3), color, dtype=np.uint8)
            np.testing.assert_array_equal(
                self.enhancer.apply_laplacian_pyramid_clarity(flat, strength=1), flat)
        noise = np.random.default_rng(17).integers(99, 102, (65, 67, 3), dtype=np.uint8)
        np.testing.assert_array_equal(
            self.enhancer.apply_laplacian_pyramid_clarity(noise, strength=0.35), noise)

    def test_pure_quality_skips_reconstruction_and_tone_filters(self):
        pipe = LocalAIEnhancerPipeline.__new__(LocalAIEnhancerPipeline)
        pipe._report_progress = Mock()
        pipe.use_re_onnx = False
        pipe.wink_enhancer = self.enhancer
        pipe._resolve_model = Mock(side_effect=AssertionError('No face reconstruction'))
        self.enhancer.apply_dehaze_and_dynamic_contrast = Mock(side_effect=AssertionError('No tone change'))
        self.enhancer.apply_deblur_deconvolution = Mock(side_effect=AssertionError('No assumed blur kernel'))
        self.enhancer.unsharp_mask = Mock(side_effect=AssertionError('No stacked sharpening'))
        for scale in (1, 2):
            out = pipe._process_image(self.edge, preset_mode='Pure Quality', upscale=scale)
            self.assertEqual(out.shape, (97 * scale, 99 * scale, 3))
        pipe._resolve_model.assert_not_called()

    def test_natural_ui_presets_keep_original_face_and_color(self):
        tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
        preset_node = next(node for node in ast.walk(tree)
                           if isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                           and isinstance(node.test.func, ast.Attribute)
                           and isinstance(node.test.func.value, ast.Name)
                           and node.test.func.value.id == 'preset_choice')
        code = compile(ast.Module(body=[preset_node], type_ignores=[]), 'preset_defaults', 'exec')
        for choice in ('Pure Quality & Sharpness', 'Natural Likeness'):
            settings = {'preset_choice': choice}
            exec(code, settings)
            for flag in ('face_restore', 'wink', 'dehaze', 'deblur', 'crystal_skin', 'doll_eye'):
                self.assertFalse(settings[f'default_{flag}'], (choice, flag))
            self.assertEqual(settings['default_sharpen'], 0)
            self.assertTrue(settings['default_super_clarity'])


class SourceBlendTests(unittest.TestCase):
    def test_public_api_forwards_blend_without_changing_model_fidelity(self):
        pipe = LocalAIEnhancerPipeline.__new__(LocalAIEnhancerPipeline)
        pipe._default_progress_callback = None
        pipe._processing_lock = threading.RLock()
        image = np.zeros((32, 32, 3), np.uint8)
        pipe._process_image = Mock(return_value=image)
        self.assertIs(pipe.process_image(image, w=0.6, source_blend=0.25), image)
        self.assertEqual(pipe._process_image.call_args.args[1], 0.6)
        self.assertEqual(pipe._process_image.call_args.kwargs['source_blend'], 0.25)

    def test_blend_endpoints_and_legacy_behavior_at_both_scales(self):
        pipe = LocalAIEnhancerPipeline.__new__(LocalAIEnhancerPipeline)
        original = np.full((64, 64, 3), 40, np.uint8)
        restored = np.full_like(original, 200)
        for scale in (1, 2):
            helper = SimpleNamespace(input_img=original, face_size=(64, 64),
                cropped_faces=[original], restored_faces=[restored], use_parse=False,
                inverse_affine_matrices=[np.array([[scale, 0, 0], [0, scale, 0]], np.float32)])
            outputs = {}
            for blend, expected in ((0, 200), (0.25, 160), (1, 40), (None, 80)):
                result = pipe.paste_faces_custom_blend(helper, upscale=scale,
                    blend_softness=0.5, w=0.75, source_blend=blend)
                outputs[blend] = result
                self.assertEqual(result.shape, (64 * scale, 64 * scale, 3))
                # Existing soft-mask arithmetic truncates the final float image;
                # even a constant interior can round down by one intensity level.
                np.testing.assert_allclose(result[32 * scale, 32 * scale], [expected] * 3, atol=1)
            explicit_legacy = pipe.paste_faces_custom_blend(helper, upscale=scale,
                blend_softness=0.5, w=0.2, source_blend=0.75)
            np.testing.assert_array_equal(outputs[None], explicit_legacy)
            np.testing.assert_array_equal(helper.restored_faces[0], restored)

    def test_invalid_blend_is_rejected_before_processing(self):
        pipe = LocalAIEnhancerPipeline.__new__(LocalAIEnhancerPipeline)
        for blend in (-0.1, 1.1, float('nan'), float('inf')):
            with self.subTest(blend=blend), self.assertRaisesRegex(ValueError, 'source_blend'):
                pipe._process_image(np.zeros((32, 32, 3), np.uint8), source_blend=blend)


class ReliabilityTests(unittest.TestCase):
    def test_lazy_pipeline_skips_model_for_pure_and_loads_on_explicit_use(self):
        for suffix, loader, attribute in (('.pth', '_get_torch_model', 'net'),
                                           ('.onnx', '_get_onnx_session', 'ort_session_cf')):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp:
                model = Path(tmp) / ('model' + suffix)
                model.write_bytes(b'x' * 128)
                with patch('pipeline.model_files'), patch.object(LocalAIEnhancerPipeline, loader) as load:
                    pipe = LocalAIEnhancerPipeline(device='cpu', model_path_override=str(model), lazy_load=True)
                    image = np.full((32, 32, 3), 80, np.uint8)
                    output = pipe.process_image(image, upscale=1, preset_mode='Pure Quality')
                    np.testing.assert_array_equal(output, image)
                    load.assert_not_called()
                    self.assertIsNone(getattr(pipe, attribute))
                    pipe._resolve_model('Auto')
                    load.assert_called_once_with(str(model.resolve()))
                    self.assertIs(getattr(pipe, attribute), load.return_value)

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
