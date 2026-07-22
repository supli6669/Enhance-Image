import os
import sys

# Ensure CodeFormer directory is on sys.path
tools_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(tools_dir)
codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")

for p in (codeformer_dir, tools_dir, project_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
import torch
from basicsr.models import build_model

def main():
    print("=== Phase 6: Stage II Transformer Network Config Verification ===")
    
    config_path = os.path.join(codeformer_dir, "options", "CodeFormer_stage2_custom.yml")
    print(f"Reading configuration from: {config_path}")
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    config["is_train"] = True
    config["num_gpu"] = 0
    config["dist"] = False
    config["dist_params"] = None
    config["path"]["root"] = codeformer_dir
    config["path"]["pretrain_network_g"] = os.path.join(project_dir, "weights", "CodeFormer", "codeformer.pth")

    
    print("Instantiating Stage II CodeFormerIdxModel architecture...")
    model = build_model(config)
    print(f"Model type '{model.__class__.__name__}' successfully built on device '{model.device}'.")
    
    print("\nSUCCESS: Phase 6 Stage II Transformer Configuration verified with exit code 0!")

if __name__ == "__main__":
    main()
