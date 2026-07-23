import os
import sys
import cv2
import torch
torch.set_num_threads(8)  # Ryzen 7735HS has 8C/16T
import yaml
import subprocess
import glob

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train CodeFormer with custom parameters.")
    parser.add_argument("--verify", action="store_true", help="Run 2 iterations for verification purposes.")
    args = parser.parse_args()

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

    # Keep the deterministic benchmark holdout out of all training phases.
    # It is intentionally ignored by Git because it may identify private data;
    # create it with tools/prepare_benchmark.py before training.
    holdout_manifest = os.path.join(project_dir, "benchmarks", "holdout_paths.txt")
    if os.path.isfile(holdout_manifest):
        for dataset in config.get("datasets", {}).values():
            dataset["exclude_manifest"] = holdout_manifest
        print(f"Excluding benchmark holdout from training: {holdout_manifest}")
    else:
        print("WARNING: benchmark holdout manifest not found; refusing quality claims for this run.")
    
    # Save original total_iter BEFORE any runtime overrides — we must restore
    # this after writing so --verify mode never permanently corrupts the file.
    original_total_iter = config.get("train", {}).get("total_iter", 20000)
        
    # Dynamically set GPU count
    config["num_gpu"] = num_gpus
    config["dist"] = False
    config["dist_params"] = None
    
    # Use multiple workers to saturate CPU cores when training on GPU; set to 0 on CPU to prevent Windows multiprocessing MemoryError/segfaults.
    if "datasets" in config:
        for phase in config["datasets"]:
            dataset = config["datasets"][phase]
            if device == "cpu":
                dataset["num_worker_per_gpu"] = 0
                # MUST be null on CPU — 'cpu' prefetch spawns multiprocessing
                # workers which cause MemoryError/segfaults on Windows (Task 8).
                dataset["prefetch_mode"] = None
            else:
                dataset["num_worker_per_gpu"] = 4
                
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
    
    if latest_state:
        print(f"\n>>> RESUME MODE: Found checkpoint at iteration {latest_state_iter}")
        print(f"    State file: {latest_state}")
        config["path"]["resume_state"] = latest_state
        config["path"]["pretrain_network_g"] = None
        if args.verify:
            config["train"]["total_iter"] = latest_state_iter + 2
            print(f"    Resuming training from iter {latest_state_iter} to {config['train']['total_iter']} (verification mode)...")
        else:
            if latest_state_iter >= config.get("train", {}).get("total_iter", 20000):
                print(f"\n>>> Training already completed ({latest_state_iter} >= {config.get('train', {}).get('total_iter', 20000)} total_iter)")
                sys.exit(0)
            print(f"    Resuming training from iter {latest_state_iter} to {config['train']['total_iter']}...")
    else:
        print("\n>>> FRESH START: No previous checkpoint found. Starting from pretrained weights.")
        if args.verify:
            config["train"]["total_iter"] = 2
            print(f"    Training from scratch to {config['train']['total_iter']} (verification mode)...")
        else:
            print(f"    Training from scratch to {config.get('train', {}).get('total_iter', 20000)}...")
    
    # 5. Write runtime config to a TEMP yml file — NEVER modify the original.
    #
    # Strategy: write all runtime overrides (num_gpu, prefetch_mode, resume_state,
    # total_iter) to a separate temp file. basicsr/train.py reads from that temp
    # file. The canonical yml with all comments is never touched again.
    #
    # This fixes two bugs permanently:
    #   (a) --verify mode used to persist total_iter=checkpoint+2 to disc, causing
    #       the next full run to stop after only 2 iterations.
    #   (b) yaml.dump was stripping all comments from the original yml every run.
    import tempfile, shutil
    tmp_fd, tmp_config_path = tempfile.mkstemp(suffix=".yml", prefix="cf_runtime_",
                                                dir=os.path.join(codeformer_dir, "options"))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Runtime config written to temp file: {os.path.basename(tmp_config_path)}")
        print(f"  device={device.upper()}, num_gpu={num_gpus}, total_iter={config['train']['total_iter']}")

        # 6. Run the training process pointing at the temp config
        train_script = os.path.join("basicsr", "train.py")
        cmd = [
            sys.executable,
            train_script,
            "-opt",
            os.path.relpath(tmp_config_path, codeformer_dir),
            "--launcher",
            "none"
        ]

        # Environment setup — CPU stability settings (Task 8 findings)
        env = os.environ.copy()
        for _k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
            env[_k] = "8"  # Match torch.set_num_threads — Ryzen 7735HS 8C/16T
        env["OPENCV_OPENCL_RUNTIME"] = "disabled"
        env["OPENCV_THREAD_LIMIT"] = "1"
        cv2.setNumThreads(0)
        env["PYTHONPATH"] = os.path.pathsep.join([codeformer_dir, env.get("PYTHONPATH", "")])

        print("\nStarting training process. Command:")
        print(" ".join(cmd))
        print(f"Working directory: {codeformer_dir}")
        print("------------------------------------------")

        try:
            result = subprocess.run(cmd, cwd=codeformer_dir, env=env, check=True)
            print("------------------------------------------")
            print("Training execution completed successfully!")
        except subprocess.CalledProcessError as e:
            print("------------------------------------------")
            print(f"Training failed with exit code: {e.returncode}")
            sys.exit(e.returncode)
    finally:
        # Always clean up the temp file after the run (success or failure)
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)

if __name__ == "__main__":
    main()
