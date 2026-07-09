import os
import shutil
import glob

def get_dir_size(path):
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except Exception:
        pass
    return total

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(project_dir, "models", "CodeFormer", "experiments")
    
    if not os.path.exists(experiments_dir):
        print("Experiments directory not found.")
        return
        
    print("=== Disk Cleanup Script ===")
    
    # Active run folder
    active_folder = "20260708_201102_CodeFormer_stage3_custom"
    active_path = os.path.join(experiments_dir, active_folder)
    
    total_freed = 0
    
    # 1. Delete old/abandoned experiment folders
    for entry in os.scandir(experiments_dir):
        if entry.is_dir() and entry.name != active_folder and entry.name != "pretrained_models":
            dir_size = get_dir_size(entry.path)
            print(f"Deleting old run: {entry.name} ({dir_size / 1024 / 1024 / 1024:.2f} GB)...")
            try:
                shutil.rmtree(entry.path)
                total_freed += dir_size
                print("Deleted.")
            except Exception as e:
                print(f"Error deleting {entry.name}: {e}")
                
    # 2. Clean intermediate checkpoints in the active folder
    if os.path.exists(active_path):
        print(f"\nCleaning intermediate checkpoints in active run: {active_folder}...")
        
        # Scan models directory
        models_dir = os.path.join(active_path, "models")
        if os.path.exists(models_dir):
            pth_files = glob.glob(os.path.join(models_dir, "*.pth"))
            # Find the highest iteration index in models
            indices = []
            for filepath in pth_files:
                basename = os.path.basename(filepath)
                if "latest" in basename:
                    continue
                # format is net_g_XX.pth or net_d_XX.pth
                try:
                    parts = basename.replace(".pth", "").split("_")
                    idx = int(parts[-1])
                    indices.append(idx)
                except ValueError:
                    continue
            
            if indices:
                highest_idx = max(indices)
                print(f"Active run highest iteration index: {highest_idx}")
                # We want to keep:
                # - files with "latest" in their name
                # - files matching highest_idx
                # - files matching highest_idx - 5 (as backup)
                keep_indices = {highest_idx, highest_idx - 5}
                
                for filepath in pth_files:
                    basename = os.path.basename(filepath)
                    if "latest" in basename:
                        continue
                    try:
                        parts = basename.replace(".pth", "").split("_")
                        idx = int(parts[-1])
                        if idx not in keep_indices:
                            file_size = os.path.getsize(filepath)
                            os.remove(filepath)
                            total_freed += file_size
                    except Exception as e:
                        print(f"Error removing {basename}: {e}")
                        
        # Scan training_states directory
        states_dir = os.path.join(active_path, "training_states")
        if os.path.exists(states_dir):
            state_files = glob.glob(os.path.join(states_dir, "*.state"))
            state_indices = []
            for filepath in state_files:
                basename = os.path.basename(filepath)
                try:
                    idx = int(basename.replace(".state", ""))
                    state_indices.append(idx)
                except ValueError:
                    continue
            
            if state_indices:
                highest_state_idx = max(state_indices)
                keep_state_indices = {highest_state_idx, highest_state_idx - 5}
                
                for filepath in state_files:
                    basename = os.path.basename(filepath)
                    try:
                        idx = int(basename.replace(".state", ""))
                        if idx not in keep_state_indices:
                            file_size = os.path.getsize(filepath)
                            os.remove(filepath)
                            total_freed += file_size
                    except Exception as e:
                        print(f"Error removing {basename}: {e}")
                        
    print(f"\nSUCCESS: Cleanup completed! Freed up {total_freed / 1024 / 1024 / 1024:.2f} GB of disk space.")

if __name__ == "__main__":
    main()
