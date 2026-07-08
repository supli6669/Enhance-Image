import os
import sys
import torch
import yaml
import subprocess

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")
    
    print("=== Custom CodeFormer Training Runner ===")
    
    # 1. Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count() if device == "cuda" else 0
    print(f"Device detected: {device.upper()}")
    print(f"Number of GPUs available: {num_gpus}")
    
    # 2. Check and prepare dataset
    dataset_dir = os.path.join(codeformer_dir, "datasets", "ffhq", "ffhq_512")
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print("Dataset directory is empty. Preparing dataset images...")
        try:
            import prepare_toy_training
            prepare_toy_training.main()
            print("Dataset preparation completed.")
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            sys.exit(1)
    else:
        print(f"Dataset found at {dataset_dir} ({len(os.listdir(dataset_dir))} images).")
        
    # 3. Update configuration file
    config_path = os.path.join(codeformer_dir, "options", "CodeFormer_stage3_custom.yml")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    print(f"Reading configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Dynamically set GPU count
    config["num_gpu"] = num_gpus
    config["dist"] = False
    config["dist_params"] = None
    
    # Force single worker on Windows / CPU to prevent pickling issues
    if "datasets" in config:
        for phase in config["datasets"]:
            dataset = config["datasets"][phase]
            dataset["num_worker_per_gpu"] = 0
            if device == "cpu":
                dataset["prefetch_mode"] = "cpu"
                
    # Update weights path if they are in the project weights folder
    project_weights_path = os.path.join(project_dir, "weights", "CodeFormer", "codeformer.pth")
    if os.path.exists(project_weights_path):
        # basicSR is relative to the running dir which is models/CodeFormer
        config["path"]["pretrain_network_g"] = "../../weights/CodeFormer/codeformer.pth"
        print(f"Configured pretrain generator path to: {config['path']['pretrain_network_g']}")
    
    # Write back the updated configuration
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Updated configuration file for device={device.upper()}, num_gpu={num_gpus}")
    
    # 4. Run the training process
    train_script = os.path.join("basicsr", "train.py")
    cmd = [
        sys.executable,
        train_script,
        "-opt",
        os.path.join("options", "CodeFormer_stage3_custom.yml"),
        "--launcher",
        "none"
    ]
    
    # Add models/CodeFormer to PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.pathsep.join([codeformer_dir, env.get("PYTHONPATH", "")])
    
    print("\nStarting training process. Command:")
    print(" ".join(cmd))
    print(f"Working directory: {codeformer_dir}")
    print("------------------------------------------")
    
    try:
        # Run subprocess under models/CodeFormer working directory
        result = subprocess.run(cmd, cwd=codeformer_dir, env=env, check=True)
        print("------------------------------------------")
        print("Training execution completed successfully!")
    except subprocess.CalledProcessError as e:
        print("------------------------------------------")
        print(f"Training failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
