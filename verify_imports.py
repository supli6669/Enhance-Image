import os
import sys
import torch

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")
    
    # Add CodeFormer path to sys.path so we can import its local basicsr and facelib
    if codeformer_dir not in sys.path:
        sys.path.insert(0, codeformer_dir)
        print(f"Added {codeformer_dir} to Python path.")
        
    try:
        # 1. Test basic imports
        print("Testing basicsr and facelib imports...")
        from basicsr.utils import img2tensor, tensor2img
        from basicsr.utils.registry import ARCH_REGISTRY
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        print("Imports successful!")
        
        # 2. Test model creation
        print("Testing CodeFormer model instantiation...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        net = ARCH_REGISTRY.get('CodeFormer')(
            dim_embd=512, 
            codebook_size=1024, 
            n_head=8, 
            n_layers=9, 
            connect_list=['32', '64', '128', '256']
        ).to(device)
        print("Model instantiated successfully!")
        
        # 3. Test weights loading
        weights_path = os.path.join(project_dir, "weights", "CodeFormer", "codeformer.pth")
        if not os.path.exists(weights_path):
            print(f"Error: Weights not found at {weights_path}")
            sys.exit(1)
            
        print(f"Loading weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location=device)
        if 'params_ema' in checkpoint:
            net.load_state_dict(checkpoint['params_ema'])
        else:
            net.load_state_dict(checkpoint['params'])
        net.eval()
        print("Weights loaded successfully! Model is ready.")
        print("\nSUCCESS: All imports and model initializations verified.")
        
    except Exception as e:
        print(f"\nFAILURE: Error occurred during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
