---
name: onnx-static-int8-quantization
description: Converting PyTorch models to ONNX static INT8 format with calibration data readers, preserving accuracy while achieving 3x-4x CPU speedup.
---

# ONNX Static INT8 Quantization & Calibration Skill

## Overview
Static INT8 quantization uses a calibration dataset to determine exact tensor activation ranges ($[\min, \max]$). This results in faster execution and lower memory bandwidth on CPU compared to dynamic quantization.

## Key Concepts
1. **Calibration Data Reader**: Implements `CalibrationDataReader` class providing 50-100 real face inputs.
2. **Quantization Mode**: `QuantFormat.QDQ` (Quantize-Dequantize) with `QuantType.QInt8`.
3. **Execution Provider**: CPUExecutionProvider optimized for AVX2/AVX512.

## Implementation Code
```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

class FaceDataReader(CalibrationDataReader):
    def __init__(self, sample_inputs):
        self.enum_data = iter([{"input": x} for x in sample_inputs])
    def get_next(self):
        return next(self.enum_data, None)

def quantize_onnx_static(in_onnx, out_onnx, sample_inputs):
    reader = FaceDataReader(sample_inputs)
    quantize_static(
        in_onnx, out_onnx, reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )
```
