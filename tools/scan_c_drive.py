import os
import sys

def get_dir_size(path, max_depth=4, current_depth=0):
    total = 0
    if current_depth > max_depth:
        return 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += get_dir_size(entry.path, max_depth, current_depth + 1)
            except Exception:
                continue
    except Exception:
        pass
    return total

def main():
    user_home = os.path.expanduser("~")
    print(f"Scanning large directories in: {user_home}...")
    
    large_dirs = []
    
    # List top level directories in user home
    try:
        entries = list(os.scandir(user_home))
    except Exception as e:
        print(f"Error reading home directory: {e}")
        return
        
    for entry in entries:
        if entry.is_dir(follow_symlinks=False):
            # Skip some standard tiny folders or system symbolic links to speed up
            if entry.name.startswith('.') and entry.name not in ['.cache', '.conda', '.docker', '.npm', '.nuget', '.vscode', '.gradle']:
                continue
            
            print(f"Checking size of {entry.name}...")
            size = get_dir_size(entry.path, max_depth=5)
            size_gb = size / (1024 * 1024 * 1024)
            if size_gb > 0.5:  # larger than 500 MB
                large_dirs.append((entry.path, size_gb))
                print(f"-> Found large dir: {entry.name} ({size_gb:.2f} GB)")
                
    print("\n=== Scan Results (Directories > 0.5 GB) ===")
    large_dirs.sort(key=lambda x: x[1], reverse=True)
    for path, size_gb in large_dirs:
        print(f"- {path}: {size_gb:.2f} GB")
        
        # If it is AppData, scan one level deeper to pinpoint the culprit
        if "AppData" in path:
            print("  Pinpointing AppData contents...")
            for sub_name in ["Local", "Roaming", "LocalLow"]:
                sub_path = os.path.join(path, sub_name)
                if os.path.exists(sub_path):
                    sub_size = get_dir_size(sub_path, max_depth=4)
                    sub_size_gb = sub_size / (1024 * 1024 * 1024)
                    if sub_size_gb > 0.5:
                        print(f"    - {sub_path}: {sub_size_gb:.2f} GB")
                        # Pinpoint Local/Roaming further
                        try:
                            for local_entry in os.scandir(sub_path):
                                if local_entry.is_dir(follow_symlinks=False):
                                    l_size = get_dir_size(local_entry.path, max_depth=3)
                                    l_size_gb = l_size / (1024 * 1024 * 1024)
                                    if l_size_gb > 0.5:
                                        print(f"      - {local_entry.path}: {l_size_gb:.2f} GB")
                        except Exception:
                            pass

if __name__ == "__main__":
    main()
