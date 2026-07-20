import os
import glob
import sys
import argparse
import cv2
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

class CodeFormerCalibrationDataReader(CalibrationDataReader):
    """Calibration Data Reader for ONNX Runtime static quantization of CodeFormer."""
    def __init__(self, calibration_folder: str, w_val: float = 0.5, max_samples: int = 5):
        super().__init__()
        valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
        self.image_paths = [
            os.path.join(calibration_folder, f) for f in os.listdir(calibration_folder)
            if f.lower().endswith(valid_exts)
        ] if os.path.exists(calibration_folder) else []

        self.w_val = np.array([w_val], dtype=np.float32)
        self.enum_data_dicts = []
        self._preprocess(max_samples)

    def _preprocess(self, max_samples: int):
        print(f"[Calib] Preprocessing up to {max_samples} calibration images...")
        count = 0
        for img_path in self.image_paths:
            if count >= max_samples:
                break
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # Normalize to [-1, 1] range matching CodeFormer inputs
            img_norm = (img_rgb - 0.5) / 0.5
            tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
            self.enum_data_dicts.append({
                "input": tensor,
                "w": self.w_val
            })
            count += 1

        if len(self.enum_data_dicts) == 0:
            print("[Calib] No calibration images found. Generating 10 synthetic calibration samples...")
            np.random.seed(42)
            for _ in range(10):
                dummy_tensor = np.random.uniform(-1.0, 1.0, (1, 3, 512, 512)).astype(np.float32)
                self.enum_data_dicts.append({
                    "input": dummy_tensor,
                    "w": self.w_val
                })

        print(f"[Calib] Prepared {len(self.enum_data_dicts)} calibration samples.")
        self.enum_data = iter(self.enum_data_dicts)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = iter(self.enum_data_dicts)

def quantize_static_model(model_path: str, output_path: str, calibration_folder: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    dr = CodeFormerCalibrationDataReader(calibration_folder)
    if dr.datasize == 0 if hasattr(dr, 'datasize') else len(dr.enum_data_dicts) == 0:
        print(f"[ERROR] Calibration folder is empty or invalid: {calibration_folder}")
        sys.exit(1)

    print(f"[Static Quant] Quantizing {model_path} -> {output_path}")
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=dr,
        quant_format=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        use_external_data_format=True
    )
    print(f"[Static Quant] Completed successfully: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static INT8 quantization for CodeFormer ONNX with calibration.")
    parser.add_argument("--model", type=str, default="weights/CodeFormer/codeformer.onnx", help="Path to input ONNX model.")
    parser.add_argument("--output", type=str, default="weights/CodeFormer/codeformer_int8_v2.onnx", help="Path to save output static quantized model.")
    parser.add_argument("--calib-dir", type=str, default="models/CodeFormer/datasets/ffhq/ffhq_512", help="Calibration images directory.")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_p = os.path.join(project_dir, args.model) if not os.path.isabs(args.model) else args.model
    out_p = os.path.join(project_dir, args.output) if not os.path.isabs(args.output) else args.output
    calib_p = os.path.join(project_dir, args.calib_dir) if not os.path.isabs(args.calib_dir) else args.calib_dir

    quantize_static_model(model_p, out_p, calib_p)
