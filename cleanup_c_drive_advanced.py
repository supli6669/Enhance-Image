import os
import shutil
import glob

def clean_directory_contents(dir_path):
    files_deleted = 0
    folders_deleted = 0
    if not os.path.exists(dir_path):
        return 0, 0
    for entry in os.scandir(dir_path):
        try:
            if entry.is_file(follow_symlinks=False):
                os.remove(entry.path)
                files_deleted += 1
            elif entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
                folders_deleted += 1
        except Exception:
            continue
    return files_deleted, folders_deleted

def main():
    print("=== Advanced C Drive Cache Cleanup ===")
    total_freed_bytes = 0
    
    # 1. VS Code C++ Extension Cache (5.18 GB)
    vscode_cpp_path = r"C:\Users\admin\AppData\Local\Microsoft\vscode-cpptools"
    if os.path.exists(vscode_cpp_path):
        print(f"Cleaning VS Code C++ extension cache...")
        # Get size before deletion
        dir_size = 0
        for root, dirs, files in os.walk(vscode_cpp_path):
            for file in files:
                try:
                    dir_size += os.path.getsize(os.path.join(root, file))
                except Exception:
                    pass
        try:
            shutil.rmtree(vscode_cpp_path)
            total_freed_bytes += dir_size
            print(f"Successfully deleted vscode-cpptools cache! (Freed ~{dir_size / 1024 / 1024 / 1024:.2f} GB)")
        except Exception as e:
            print(f"Failed to completely delete vscode-cpptools cache: {e}")
            
    # 2. Large Installer Files in Downloads (~1.6 GB)
    downloads_path = r"C:\Users\admin\Downloads"
    if os.path.exists(downloads_path):
        print("Cleaning large installer (.exe) files in Downloads (> 50 MB)...")
        deleted_count = 0
        deleted_size = 0
        for entry in os.scandir(downloads_path):
            if entry.is_file() and entry.name.lower().endswith(".exe"):
                try:
                    size = entry.stat().st_size
                    if size > 50 * 1024 * 1024:  # > 50 MB
                        os.remove(entry.path)
                        deleted_count += 1
                        deleted_size += size
                except Exception as e:
                    print(f"Failed to remove {entry.name}: {e}")
        total_freed_bytes += deleted_size
        print(f"Deleted {deleted_count} setup files in Downloads. (Freed ~{deleted_size / 1024 / 1024:.2f} MB)")
        
    # 3. Microsoft Edge Cache (~2 GB)
    edge_caches = [
        r"C:\Users\admin\AppData\Local\Microsoft\Edge\User Data\Default\Cache",
        r"C:\Users\admin\AppData\Local\Microsoft\Edge\User Data\Default\Code Cache"
    ]
    for cache_path in edge_caches:
        if os.path.exists(cache_path):
            print(f"Cleaning Edge browser cache: {os.path.basename(cache_path)}...")
            # Calculate size
            dir_size = 0
            for root, dirs, files in os.walk(cache_path):
                for file in files:
                    try:
                        dir_size += os.path.getsize(os.path.join(root, file))
                    except Exception:
                        pass
            files_del, folders_del = clean_directory_contents(cache_path)
            total_freed_bytes += dir_size
            print(f"Edge cache cleaned. Removed {files_del} files, {folders_del} folders. (Freed ~{dir_size / 1024 / 1024:.2f} MB)")
            
    # 4. Google Chrome Cache (Optional)
    chrome_caches = [
        r"C:\Users\admin\AppData\Local\Google\Chrome\User Data\Default\Cache",
        r"C:\Users\admin\AppData\Local\Google\Chrome\User Data\Default\Code Cache"
    ]
    for cache_path in chrome_caches:
        if os.path.exists(cache_path):
            print(f"Cleaning Chrome browser cache: {os.path.basename(cache_path)}...")
            dir_size = 0
            for root, dirs, files in os.walk(cache_path):
                for file in files:
                    try:
                        dir_size += os.path.getsize(os.path.join(root, file))
                    except Exception:
                        pass
            files_del, folders_del = clean_directory_contents(cache_path)
            total_freed_bytes += dir_size
            print(f"Chrome cache cleaned. (Freed ~{dir_size / 1024 / 1024:.2f} MB)")
            
    # 5. Discord Cache (Optional)
    discord_caches = [
        r"C:\Users\admin\AppData\Roaming\discord\Cache",
        r"C:\Users\admin\AppData\Roaming\discord\Code Cache",
        r"C:\Users\admin\AppData\Roaming\discord\GPUCache"
    ]
    for cache_path in discord_caches:
        if os.path.exists(cache_path):
            print(f"Cleaning Discord cache: {os.path.basename(cache_path)}...")
            dir_size = 0
            for root, dirs, files in os.walk(cache_path):
                for file in files:
                    try:
                        dir_size += os.path.getsize(os.path.join(root, file))
                    except Exception:
                        pass
            files_del, folders_del = clean_directory_contents(cache_path)
            total_freed_bytes += dir_size
            print(f"Discord cache cleaned. (Freed ~{dir_size / 1024 / 1024:.2f} MB)")
            
    print(f"\nSUCCESS: Advanced C drive cleanup completed! Total freed up: {total_freed_bytes / 1024 / 1024 / 1024:.2f} GB.")

if __name__ == "__main__":
    main()
