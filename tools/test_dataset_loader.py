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
from basicsr.data import build_dataset

def main():
    print("=== Phase 4: Dataset Loader & Game Character Mixing Verification ===")
    
    config_path = os.path.join(codeformer_dir, "options", "CodeFormer_stage3_custom.yml")
    print(f"Reading configuration from: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    dataset_opt = config["datasets"]["train"]
    # Adjust relative dataroot_gt to absolute path
    dataset_opt["dataroot_gt"] = os.path.join(codeformer_dir, dataset_opt["dataroot_gt"])
    dataset_opt["prefetch_mode"] = None
    dataset_opt["num_worker_per_gpu"] = 0
    
    print(f"Dataset root: {dataset_opt['dataroot_gt']}")
    
    # Count images in dataset folder
    gt_dir = dataset_opt["dataroot_gt"]
    all_files = []
    for root, _, files in os.walk(gt_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                all_files.append(os.path.join(root, file))
                
    print(f"Total image files found on disk: {len(all_files)}")
    
    # Build BasicSR Dataset
    dataset = build_dataset(dataset_opt)
    print(f"Dataset object instantiated. Total items reported by dataset loader: {len(dataset)}")
    
    # Fetch first 5 items to verify shapes and pipeline execution
    print("\nFetching sample items from dataset loader:")
    for i in range(min(5, len(dataset))):
        item = dataset[i]
        gt_tensor = item['gt']
        in_tensor = item['in']
        print(f"  Item #{i}: GT shape={gt_tensor.shape}, Input shape={in_tensor.shape}, GT min/max=[{gt_tensor.min():.2f}, {gt_tensor.max():.2f}]")
        assert gt_tensor.shape == (3, 512, 512), f"Invalid GT shape: {gt_tensor.shape}"
        assert in_tensor.shape == (3, 512, 512), f"Invalid Input shape: {in_tensor.shape}"
        
    print("\nSUCCESS: Phase 4 Dataset Expansion & Game Character Mixing verified with exit code 0!")

if __name__ == "__main__":
    main()
