import os
import requests
import yaml

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    codeformer_dir = os.path.join(project_dir, "models", "CodeFormer")
    
    # 1. Create target dataset directory
    dataset_dir = os.path.join(codeformer_dir, "datasets", "ffhq", "ffhq_512")
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Created dataset directory: {dataset_dir}")
    
    # 2. Download toy dataset (30 face images from Unsplash)
    unsplash_ids = [
        "1544005313-94ddf0286df2", "1506794778202-cad84cf45f1d", "1534528741775-53994a69daeb",
        "1507003211169-0a1dd7228f2d", "1522075469751-3a6694fb2f61", "1544717305-2782549b5136",
        "1554151228-14d9def656e4", "1531746020798-e6953c6e8e04", "1500648767791-00dcc994a43e",
        "1508214751196-bcfd4ca60f91", "1494790108377-be9c29b29330", "1517841905240-472988babdf9",
        "1539571696357-5a69c17a67c6", "1438761681033-6461ffad8d80", "1524504388940-b1c1722653e1",
        "1519085360753-af0119f7cbe7", "1491528920044-4531310b6531", "1503023344727-8982f00d2947",
        "1534308983496-4fabb1a015ee", "1542206395-9feb3edaa68d", "1501196354995-cbb51c65aaea",
        "1506863530036-1775a06bfa37", "1508214751196-bcfd4ca60f91", "1513956589380-bad6acb9b9d4",
        "1519345182560-3f2917c472ef", "1520155707334-757655122b38", "1530577197743-7adf14294584",
        "1531123897727-8f129e1688ce", "1539571696357-5a69c17a67c6", "1548142813-c348350df52b"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print("Downloading 30 face images from Unsplash (512x512 cropped)...")
    for i, img_id in enumerate(unsplash_ids):
        filename = f"{i:05d}.png"
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            print(f"Skipping {filename} (already exists)")
            continue
            
        url = f"https://images.unsplash.com/photo-{img_id}?w=512&h=512&fit=crop&q=80"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(r.content)
                print(f"Downloaded {filename}")
            else:
                print(f"Failed to download {filename} (Status: {r.status_code})")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            
    # 3. Create local toy configurations for Stages I, II, and III
    configs_to_make = [
        ("VQGAN_512_ds32_nearest_stage1.yml", "VQGAN_toy.yml"),
        ("CodeFormer_stage2.yml", "CodeFormer_stage2_toy.yml"),
        ("CodeFormer_stage3.yml", "CodeFormer_stage3_toy.yml")
    ]
    
    for orig_name, toy_name in configs_to_make:
        orig_path = os.path.join(codeformer_dir, "options", orig_name)
        toy_path = os.path.join(codeformer_dir, "options", toy_name)
        
        if not os.path.exists(orig_path):
            print(f"Error: Original config {orig_name} not found!")
            continue
            
        with open(orig_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        # Modify for local CPU toy training
        config["num_gpu"] = 0
        config["dist_params"] = None
        config["dist"] = False
        
        if "datasets" in config:
            for phase in config["datasets"]:
                dataset = config["datasets"][phase]
                dataset["num_worker_per_gpu"] = 0  # No multiprocessing on CPU (avoid pickling/Windows issues)
                dataset["batch_size_per_gpu"] = 1
                dataset["dataset_enlarge_ratio"] = 1  # Don't enlarge dataset, keep it small
                if "prefetch_mode" in dataset:
                    dataset["prefetch_mode"] = "cpu"
                    
        if "train" in config:
            config["train"]["total_iter"] = 5  # Train for only 5 iterations to test
            
        if "logger" in config:
            config["logger"]["print_freq"] = 1
            config["logger"]["save_checkpoint_freq"] = 5
            config["logger"]["use_tb_logger"] = False
            config["logger"]["wandb"] = None
            
        with open(toy_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Created local toy config: {toy_path}")
        
    print("\nSUCCESS: Toy training setup complete!")

if __name__ == "__main__":
    main()
