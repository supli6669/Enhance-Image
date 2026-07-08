import os
import sys
import torch
import yaml
import subprocess
import glob

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
    
    # 4. Auto-detect resume state from the latest experiment checkpoint
    experiments_dir = os.path.join(codeformer_dir, "experiments")
    latest_state = None
    latest_state_iter = 0
    
    # Search for existing experiment directories matching our config name pattern
    exp_pattern = os.path.join(experiments_dir, "*_CodeFormer_stage3_custom", "training_states", "*.state")
    state_files = glob.glob(exp_pattern)
    
    for state_file in state_files:
        basename = os.path.basename(state_file)
        try:
            iter_num = int(basename.replace(".state", ""))
            if iter_num > latest_state_iter:
                latest_state_iter = iter_num
                latest_state = state_file
        except ValueError:
            continue
    
    if latest_state and latest_state_iter < config.get("train", {}).get("total_iter", 50):
        print(f"\n>>> RESUME MODE: Found checkpoint at iteration {latest_state_iter}")
        print(f"    State file: {latest_state}")
        config["path"]["resume_state"] = latest_state
        # When resuming, we don't need pretrain_network_g as it will be loaded from state
        config["path"]["pretrain_network_g"] = None
        print(f"    Resuming training from iter {latest_state_iter} to {config['train']['total_iter']}...")
    elif latest_state and latest_state_iter >= config.get("train", {}).get("total_iter", 50):
        print(f"\n>>> Training already completed ({latest_state_iter} >= {config['train']['total_iter']} total_iter)")
        print("    To train more iterations, increase total_iter in the config.")
        sys.exit(0)
    else:
        print("\n>>> FRESH START: No previous checkpoint found. Starting from pretrained weights.")
    
    # Write back the updated configuration
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Updated configuration file for device={device.upper()}, num_gpu={num_gpus}")
    
    # 5. Run the training process
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
