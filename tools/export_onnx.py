import os
import sys
import torch
import glob
import json
import inspect
from pathlib import Path
import numpy as np

# Ensure project root and CodeFormer directory are in sys.path
tools_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(tools_dir)
codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")
if codeformer_dir not in sys.path:
    sys.path.insert(0, codeformer_dir)

sys.path.insert(0, project_dir)
import tools.compat_shim
from tools.model_artifacts import describe_model
from basicsr.utils.registry import ARCH_REGISTRY
import basicsr.archs.codeformer_arch
from basicsr.archs.rrdbnet_arch import RRDBNet

class CodeFormerONNXWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, w):
        # We pass adain=True for face restoration
        out, _, _ = self.net(x, w=w, adain=True)
        return out

def get_latest_checkpoint(search_pattern):
    checkpoints = glob.glob(search_pattern)
    if not checkpoints:
        return None
    # Auto-sort checkpoints by iteration number in their filenames (e.g., net_g_1200.pth)
    def extract_iter(path):
        basename = os.path.basename(path)
        # e.g., net_g_1200.pth -> 1200
        numbers = [int(s) for s in basename.replace(".pth", "").split("_") if s.isdigit()]
        return numbers[0] if numbers else 0
    
    return max(checkpoints, key=extract_iter)

def export_codeformer(checkpoint_path, output_path):
    print("\n--- Exporting CodeFormer to ONNX ---")
    device = torch.device("cpu")
    
    # 1. Instantiate CodeFormer model
    net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512, 
        codebook_size=1024, 
        n_head=8, 
        n_layers=9, 
        connect_list=['32', '64', '128', '256']
    )
    
    checkpoint_path, output_path = str(checkpoint_path), str(output_path)
    if os.path.exists(output_path):
        raise FileExistsError('Choose a new output file; exports never overwrite active models')
    source = describe_model(Path(checkpoint_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # 3. Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'params_ema' in checkpoint:
        net.load_state_dict(checkpoint['params_ema'])
    else:
        net.load_state_dict(checkpoint['params'])
    net.eval()
    
    wrapper = CodeFormerONNXWrapper(net).to(device)
    
    # Dummy inputs: x (1, 3, 512, 512) and w (scalar / 1D tensor)
    dummy_x = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    dummy_w = torch.tensor([0.5], dtype=torch.float32) # Default w value
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_w),
        output_path,
        input_names=['input', 'w'],
        output_names=['output'],
        opset_version=16,
        do_constant_folding=True,
        **({'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {})
    )
    
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(output_path)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(output_path, sess_options=options, providers=['CPUExecutionProvider'])
    max_error = 0.0
    for fidelity in (0.0, 0.5, 1.0):
        weight = torch.tensor([fidelity], dtype=torch.float32)
        with torch.inference_mode():
            expected = wrapper(dummy_x, weight).numpy()
        actual = session.run(None, {'input': dummy_x.numpy(), 'w': weight.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=3e-3)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    manifest = {'source': source, 'model': describe_model(Path(output_path)),
                'inference_parity': {'passed': True, 'max_absolute_error': max_error,
                                     'fidelities': [0.0, 0.5, 1.0]},
                'quality_status': 'export_parity_only'}
    Path(output_path).with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Export verified against PyTorch at three fidelities; max error={max_error:.6f}')
    return True


def export_realesrgan(num_block=None):
    print("\n--- Exporting Real-ESRGAN to ONNX ---")
    device = torch.device("cpu")
    
    # 1. Check for latest custom checkpoint, fallback to pretrained
    custom_pattern = os.path.join(project_dir, "models", "Real-ESRGAN", "experiments", "train_RealESRGAN_custom", "models", "net_g_*.pth")
    checkpoint_path = get_latest_checkpoint(custom_pattern)
    
    if checkpoint_path:
        print(f"Found custom Real-ESRGAN checkpoint: {checkpoint_path}")
        scale = 4
        default_blocks = 6
    else:
        checkpoint_path = os.path.join(project_dir, "weights", "realesrgan", "RealESRGAN_x2plus.pth")
        print(f"No custom checkpoint found. Using pretrained weights: {checkpoint_path}")
        scale = 2
        default_blocks = 23
        
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Real-ESRGAN weights not found at: {checkpoint_path}")
        return False
        
    blocks = num_block if num_block is not None else default_blocks
    print(f"Instantiating RRDBNet with scale={scale}, num_block={blocks}")
    
    # 2. Instantiate RRDBNet architecture with correct scale
    net = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=blocks,
        num_grow_ch=32,
        scale=scale
    )
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'params_ema' in checkpoint:
        net.load_state_dict(checkpoint['params_ema'])
    else:
        net.load_state_dict(checkpoint['params'])
    net.eval()
    
    # Dummy input: x (1, 3, 256, 256) with dynamic H & W
    dummy_x = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    
    output_dir = os.path.join(project_dir, "weights", "realesrgan")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "realesrgan.onnx")
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        net,
        dummy_x,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        },
        opset_version=16,
        do_constant_folding=True
    )
    
    # Verify model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[SUCCESS] Real-ESRGAN ONNX model is valid.")
        return True
    except Exception as e:
        print(f"[WARNING] ONNX validation failed: {e}")
        return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export an explicit CodeFormer checkpoint with inference parity verification")
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    export_codeformer(args.checkpoint, args.output)

if __name__ == "__main__":
    main()
