import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights_dir = os.path.join(project_dir, "weights", "CodeFormer")
calib_dir = os.path.join(project_dir, "models", "CodeFormer", "datasets", "ffhq", "ffhq_512")

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

import gc

def benchmark_model(model_name, model_path, test_images, ort_opts):
    if not os.path.exists(model_path):
        print(f"[{model_name}] Model file not found: {model_path}")
        return None
        
    session = ort.InferenceSession(model_path, sess_options=ort_opts, providers=['CPUExecutionProvider'])
    input_names = [inp.name for inp in session.get_inputs()]
    
    times = []
    outputs = []
    
    # Warmup
    dummy_input = np.zeros((1, 3, 512, 512), dtype=np.float32)
    w = np.array([0.5], dtype=np.float32)
    session.run(None, {input_names[0]: dummy_input, input_names[1]: w})
    
    for img_path in test_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - 0.5) / 0.5
        tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        
        t0 = time.time()
        out = session.run(None, {input_names[0]: tensor, input_names[1]: w})[0]
        t1 = time.time()
        times.append(t1 - t0)
        outputs.append(out)
        
    del session
    gc.collect()
    
    avg_time = np.mean(times) if times else 0
    return avg_time, outputs

def main():
    print("=== ONNX Quantization Benchmark (P5.4) ===")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    test_files = [os.path.join(calib_dir, f) for f in os.listdir(calib_dir)[:10] if f.endswith('.png')]
    print(f"Benchmarking on {len(test_files)} sample face crops...")
    
    models = {
        "FP32 Baseline": os.path.join(weights_dir, "codeformer.onnx"),
        "Dynamic INT8": os.path.join(weights_dir, "codeformer_int8.onnx"),
        "Static INT8 (v2)": os.path.join(weights_dir, "codeformer_int8_v2.onnx")
    }
    
    results = {}
    baseline_outputs = None
    
    for name, path in models.items():
        res = benchmark_model(name, path, test_files, opts)
        if res is not None:
            avg_time, outputs = res
            results[name] = {"time": avg_time, "outputs": outputs}
            if name == "FP32 Baseline":
                baseline_outputs = outputs
                
    print("\n-----------------------------------------------------------")
    print(f"{'Model Variant':<20} | {'Avg Latency (ms)':<18} | {'PSNR vs FP32 (dB)':<18}")
    print("-----------------------------------------------------------")
    
    for name, data in results.items():
        avg_ms = data["time"] * 1000
        if name == "FP32 Baseline":
            psnr_str = "N/A (Baseline)"
        else:
            psnrs = []
            for b_out, q_out in zip(baseline_outputs, data["outputs"]):
                b_img = ((np.squeeze(b_out).transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                q_img = ((np.squeeze(q_out).transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                psnrs.append(calculate_psnr(b_img, q_img))
            mean_psnr = np.mean(psnrs)
            psnr_str = f"{mean_psnr:.2f} dB"
            
        print(f"{name:<20} | {avg_ms:<18.2f} | {psnr_str:<18}")
    print("-----------------------------------------------------------\n")

if __name__ == "__main__":
    main()
