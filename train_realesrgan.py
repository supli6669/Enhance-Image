"""
Train Real-ESRGAN for general image enhancement (landscape, anime, scenery).
This script:
  1. Clones Real-ESRGAN repo if not present
  2. Prepares a training config pointing to our crawled dataset
  3. Launches the training process
"""
import os
import sys
import subprocess
import shutil
import yaml
import glob

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REALESRGAN_DIR = os.path.join(PROJECT_DIR, "models", "Real-ESRGAN")

# Dataset paths (populated by crawl_datasets.py)
DATASET_BASE = os.path.join(
    PROJECT_DIR, "models", "CodeFormer", "datasets", "ffhq", "ffhq_512"
)
LANDSCAPE_DIR = os.path.join(DATASET_BASE, "landscapes")
ANIME_DIR = os.path.join(DATASET_BASE, "anime")
FACE_DIR = os.path.join(DATASET_BASE, "faces")
STARRAIL_DIR = os.path.join(DATASET_BASE, "starrail")
BLUEARCHIVE_DIR = os.path.join(DATASET_BASE, "bluearchive")

# Combined GT (high-quality) folder for Real-ESRGAN training
REALESRGAN_GT_DIR = os.path.join(PROJECT_DIR, "datasets", "realesrgan_gt")


def clone_realesrgan():
    """Clone Real-ESRGAN repository if not already present."""
    if os.path.exists(REALESRGAN_DIR):
        print("[OK] Real-ESRGAN repo already present.")
        return True
    print("Cloning Real-ESRGAN repository...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/xinntao/Real-ESRGAN.git", REALESRGAN_DIR],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[FAIL] Failed to clone Real-ESRGAN.")
        return False
    print("[OK] Real-ESRGAN cloned successfully.")
    return True


def install_realesrgan_deps():
    """Install Real-ESRGAN dependencies."""
    print("Installing Real-ESRGAN dependencies...")
    # Skip the repo's requirements.txt — it pins an old basicsr that fails to
    # build on Python 3.13.  We already have a working basicsr installed via
    # the CodeFormer setup, so only install the extras Real-ESRGAN needs.
    extras = ["facexlib", "gfpgan"]
    for pkg in extras:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  [OK] {pkg} installed.")
        else:
            print(f"  [warn] {pkg} install failed (may already be present): {result.stderr[-200:]}")
    print("[OK] Dependencies ready.")


def prepare_gt_dataset():
    """Combine landscape + anime + face images into a single GT folder."""
    os.makedirs(REALESRGAN_GT_DIR, exist_ok=True)

    total_copied = 0
    sources = {
        "landscape": LANDSCAPE_DIR,
        "anime": ANIME_DIR,
        "face": FACE_DIR,
        "starrail": STARRAIL_DIR,
        "bluearchive": BLUEARCHIVE_DIR,
    }

    for category, src_dir in sources.items():
        if not os.path.exists(src_dir):
            print(f"  [skip] {category} dir not found: {src_dir}")
            continue
        images = glob.glob(os.path.join(src_dir, "*.png"))
        print(f"  Copying {len(images)} {category} images...")
        for img_path in images:
            dest = os.path.join(REALESRGAN_GT_DIR, os.path.basename(img_path))
            if not os.path.exists(dest):
                shutil.copy2(img_path, dest)
                total_copied += 1

    # Also copy root-level face images (original FFHQ)
    root_images = glob.glob(os.path.join(DATASET_BASE, "*.png"))
    print(f"  Copying {len(root_images)} root FFHQ face images...")
    for img_path in root_images:
        dest = os.path.join(REALESRGAN_GT_DIR, os.path.basename(img_path))
        if not os.path.exists(dest):
            shutil.copy2(img_path, dest)
            total_copied += 1

    total = len([f for f in os.listdir(REALESRGAN_GT_DIR) if f.endswith(".png")])
    print(f"[OK] GT dataset ready: {total} images (copied {total_copied} new)")

    # Generate meta_info.txt  (format: "filename.png 1" per line)
    meta_info_path = os.path.join(REALESRGAN_GT_DIR, "meta_info.txt")
    all_pngs = sorted([f for f in os.listdir(REALESRGAN_GT_DIR) if f.endswith(".png")])
    with open(meta_info_path, "w", encoding="utf-8") as mf:
        for fname in all_pngs:
            mf.write(f"{fname} 1\n")
    print(f"[OK] meta_info.txt written: {len(all_pngs)} entries -> {meta_info_path}")
    return total


def create_training_config(num_images: int) -> str:
    """Create a Real-ESRGAN training YAML config."""
    config_dir = os.path.join(REALESRGAN_DIR, "options")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "train_realesrgan_custom.yml")

    # Relative path from Real-ESRGAN dir to GT dataset
    gt_path_rel = os.path.relpath(REALESRGAN_GT_DIR, REALESRGAN_DIR).replace("\\", "/")

    config = {
        "name": "train_RealESRGAN_custom",
        "model_type": "RealESRGANModel",
        "scale": 4,
        "num_gpu": 0,
        "manual_seed": 0,

        # Top-level options for synthesizing training data in RealESRGANModel
        "l1_gt_usm": True,
        "percep_gt_usm": True,
        "gan_gt_usm": False,

        # the first degradation process
        "resize_prob": [0.2, 0.7, 0.1],
        "resize_range": [0.15, 1.5],
        "gaussian_noise_prob": 0.5,
        "noise_range": [1, 30],
        "poisson_scale_range": [0.05, 3.0],
        "gray_noise_prob": 0.4,
        "jpeg_range": [30, 95],

        # the second degradation process
        "second_blur_prob": 0.8,
        "resize_prob2": [0.3, 0.4, 0.3],
        "resize_range2": [0.3, 1.2],
        "gaussian_noise_prob2": 0.5,
        "noise_range2": [1, 25],
        "poisson_scale_range2": [0.05, 2.5],
        "gray_noise_prob2": 0.4,
        "jpeg_range2": [30, 95],

        "gt_size": 256,
        "queue_size": 20,  # small queue size for CPU/single worker

        "datasets": {
            "train": {
                "name": "CustomMixedDataset",
                "type": "RealESRGANDataset",
                "dataroot_gt": gt_path_rel,
                "meta_info": os.path.join(REALESRGAN_GT_DIR, "meta_info.txt").replace("\\", "/"),
                "io_backend": {"type": "disk"},
                "gt_size": 256,
                "use_hflip": True,
                "use_rot": False,
                "blur_kernel_size": 21,
                "kernel_list": ["iso", "aniso", "generalized_iso", "generalized_aniso",
                                "plateau_iso", "plateau_aniso"],
                "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                "sinc_prob": 0.1,
                "blur_sigma": [0.2, 3.0],
                "betag_range": [0.5, 4.0],
                "betap_range": [1, 2.0],
                "blur_kernel_size2": 21,
                "kernel_list2": ["iso", "aniso", "generalized_iso", "generalized_aniso",
                                 "plateau_iso", "plateau_aniso"],
                "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                "sinc_prob2": 0.1,
                "blur_sigma2": [0.2, 1.5],
                "betag_range2": [0.5, 4.0],
                "betap_range2": [1, 2.0],
                "final_sinc_prob": 0.8,
                "num_worker_per_gpu": 0,
                "batch_size_per_gpu": 2,
                "dataset_enlarge_ratio": 1,
                "prefetch_mode": "cpu",
                "num_prefetch_queue": 1,
            }
        },
        "network_g": {
            "type": "RRDBNet",
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 23,
            "num_grow_ch": 32,
            "scale": 4,
        },
        "network_d": {
            "type": "UNetDiscriminatorSN",
            "num_in_ch": 3,
            "num_feat": 64,
            "skip_connection": True,
        },
        "path": {
            "pretrain_network_g": None,
            "param_key_g": "params_ema",
            "strict_load_g": False,
            "resume_state": None,
        },
        "train": {
            "ema_decay": 0.999,
            "optim_g": {
                "type": "Adam",
                "lr": 1.0e-4,
                "weight_decay": 0,
                "betas": [0.9, 0.99],
            },
            "optim_d": {
                "type": "Adam",
                "lr": 1.0e-4,
                "weight_decay": 0,
                "betas": [0.9, 0.99],
            },
            "scheduler": {
                "type": "MultiStepLR",
                "milestones": [400000],
                "gamma": 0.5,
            },
            "total_iter": 1000,
            "warmup_iter": -1,
            "pixel_opt": {
                "type": "L1Loss",
                "loss_weight": 1.0,
                "reduction": "mean",
            },
            "perceptual_opt": {
                "type": "PerceptualLoss",
                "layer_weights": {"conv1_2": 0.1, "conv2_2": 0.1, "conv3_4": 1.0,
                                  "conv4_4": 1.0, "conv5_4": 1.0},
                "vgg_type": "vgg19",
                "use_input_norm": True,
                "range_norm": False,
                "perceptual_weight": 1.0,
                "style_weight": 0.0,
                "criterion": "l1",
            },
            "gan_opt": {
                "type": "GANLoss",
                "gan_type": "vanilla",
                "real_label_val": 1.0,
                "fake_label_val": 0.0,
                "loss_weight": 1.0e-1,
            },
            "net_d_iters": 1,
            "net_d_start_iter": 0,
        },
        "val": {
            "val_freq": 500,
            "save_img": True,
        },
        "logger": {
            "print_freq": 1,
            "save_checkpoint_freq": 50,
            "use_tb_logger": False,
            "wandb": {"project": None},
        },
        "dist_params": None,
        "dist": False,
        "find_unused_parameters": False,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[OK] Config written: {config_path}")
    return config_path


def find_latest_state() -> str | None:
    """Find the latest training state checkpoint."""
    pattern = os.path.join(
        REALESRGAN_DIR, "experiments", "*train_RealESRGAN_custom*",
        "training_states", "*.state"
    )
    states = glob.glob(pattern)
    if not states:
        return None
    latest = max(states, key=lambda p: int(os.path.basename(p).replace(".state", "")))
    return latest


def main():
    print("=" * 55)
    print("  Real-ESRGAN Custom Training Runner")
    print("  Target: Landscape + Anime + Face enhancement")
    print("=" * 55)

    # 1. Clone repo
    if not clone_realesrgan():
        sys.exit(1)

    # 2. Install deps
    install_realesrgan_deps()

    # 3. Prepare combined GT dataset
    print("\n[Step 3] Preparing GT dataset...")
    num_images = prepare_gt_dataset()
    if num_images == 0:
        print("[FAIL] No images found in dataset directories. Run crawl_datasets.py first.")
        sys.exit(1)

    # 4. Create training config
    print("\n[Step 4] Creating training config...")
    config_path = create_training_config(num_images)

    # 5. Check for resume state
    latest_state = find_latest_state()
    if latest_state:
        iter_num = int(os.path.basename(latest_state).replace(".state", ""))
        print(f"\n>>> RESUME MODE: Found checkpoint at iteration {iter_num}")
        # Patch config to resume
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["path"]["resume_state"] = latest_state
        cfg["path"]["pretrain_network_g"] = None
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    else:
        print("\n>>> FRESH START: No previous checkpoint. Starting from scratch.")

    # 6. Run training
    train_script = os.path.join("realesrgan", "train.py")
    cmd = [
        sys.executable,
        train_script,
        "-opt", os.path.join("options", "train_realesrgan_custom.yml"),
        "--launcher", "none",
    ]

    env = os.environ.copy()
    # Real-ESRGAN's train.py imports from basicsr.
    # CodeFormer's basicsr lacks degradations module, so we use the full
    # BasicSR source cloned at D:\Temp\BasicSR_src (no install needed).
    basicsr_src = r"D:\Temp\BasicSR_src"
    codeformer_dir = os.path.join(PROJECT_DIR, "models", "CodeFormer")
    env["PYTHONPATH"] = os.path.pathsep.join([
        REALESRGAN_DIR,
        basicsr_src,       # full BasicSR with degradations
        codeformer_dir,    # fallback
        env.get("PYTHONPATH", ""),
    ])

    print(f"\nStarting Real-ESRGAN training...")
    print(f"Working directory: {REALESRGAN_DIR}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 55)

    try:
        subprocess.run(cmd, cwd=REALESRGAN_DIR, env=env, check=True)
        print("-" * 55)
        print("[OK] Real-ESRGAN training completed!")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Training failed with exit code: {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
