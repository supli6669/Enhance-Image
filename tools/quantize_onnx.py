import os
import argparse
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(input_path: str, output_path: str = None, per_channel: bool = False):
    """Quantize an ONNX model to INT8.

    Args:
        input_path: Path to the original ONNX model.
        output_path: Destination path. If None, will create a file with suffix `_int8.onnx`.
        per_channel: Use per-channel quantization if True (requires onnxruntime >= 1.13).
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"ONNX model not found: {input_path}")
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_int8{ext}"
    print(f"[Quant] Loading model from {input_path}")
    model = onnx.load(input_path)
    onnx.checker.check_model(model)
    print(f"[Quant] Starting dynamic INT8 quantization (per_channel={per_channel})")
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    print(f"[Quant] Quantized model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize an ONNX model to INT8 for faster CPU inference.")
    parser.add_argument("model_path", type=str, help="Path to the original ONNX model file.")
    parser.add_argument("--output", type=str, default=None, help="Output path for the quantized model.")
    parser.add_argument("--per-channel", action="store_true", help="Enable per‑channel quantization (may improve accuracy).")
    args = parser.parse_args()
    quantize_model(args.model_path, args.output, per_channel=args.per_channel)
