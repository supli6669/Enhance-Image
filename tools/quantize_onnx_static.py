"""Static QDQ INT8 calibration with explicit, real image inputs and provenance."""
import argparse
import json
import random
from pathlib import Path
import sys
import cv2
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_artifacts import describe_model, model_files


class CodeFormerCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_folder, w_val=0.5, max_samples=100, min_samples=32, image_manifest=None):
        root = Path(calibration_folder)
        paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'})
        if image_manifest is not None:
            root = root.resolve()
            paths = [(root / name).resolve() for name in Path(image_manifest).read_text(encoding='utf-8').splitlines() if name.strip()]
            if any(not path.is_relative_to(root) or not path.is_file() for path in paths):
                raise ValueError('Invalid calibration manifest path')
        random.Random(42).shuffle(paths)
        self.image_paths = paths[:max_samples]
        if len(self.image_paths) < min_samples:
            raise ValueError(f'Need at least {min_samples} real calibration images; found {len(self.image_paths)}')
        self.w_val = np.array([w_val], dtype=np.float32)
        self.rewind()

    def get_next(self):
        path = next(self.iterator, None)
        if path is None:
            return None
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable calibration image: {path}')
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return {'input': image.transpose(2, 0, 1)[None], 'w': self.w_val}

    def rewind(self):
        self.iterator = iter(self.image_paths)


def quantize_static_model(model_path, output_path, calibration_folder, max_samples=100, image_manifest=None):
    output = Path(output_path)
    if output.exists() or output.with_name(output.name + '.data').exists():
        raise FileExistsError('Choose a new output filename; active model files are never overwritten')
    source = describe_model(Path(model_path))
    reader = CodeFormerCalibrationDataReader(calibration_folder, max_samples=max_samples, image_manifest=image_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(model_input=str(model_path), model_output=str(output),
                    calibration_data_reader=reader, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    use_external_data_format=True)
    model_files(output)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(output), sess_options=options, providers=['CPUExecutionProvider'])
    reader.rewind()
    result = session.run(None, reader.get_next())[0]
    if result.shape != (1, 3, 512, 512) or not np.isfinite(result).all():
        raise ValueError('Quantized model returned invalid inference output')
    metadata = {'quantization': 'static_QDQ_INT8', 'source': source,
                'calibration_count': len(reader.image_paths), 'seed': 42,
                'model': describe_model(output), 'quality_status': 'requires_FP32_comparison'}
    output.with_suffix('.manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Static INT8 verified on {len(reader.image_paths)} calibration inputs: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calib-dir', type=Path, required=True, help='Representative aligned face crops, not benchmark holdout')
    parser.add_argument('--calib-manifest', type=Path, required=True, help='Reviewed training split manifest; never use benchmark holdout')
    parser.add_argument('--max-samples', type=int, default=100)
    args = parser.parse_args()
    quantize_static_model(args.model, args.output, args.calib_dir, args.max_samples, args.calib_manifest)
