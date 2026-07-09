import os
import shutil

def main():
    print("=== C Drive Cache Cleanup ===")
    
    # 1. Spotify Cache Data (8.09 GB)
    spotify_data = r"C:\Users\admin\AppData\Local\Spotify\Data"
    if os.path.exists(spotify_data):
        print("Cleaning Spotify local streaming cache (Data)...")
        try:
            shutil.rmtree(spotify_data)
            print("Successfully deleted Spotify local cache!")
        except Exception as e:
            print(f"Failed to delete Spotify cache (some files may be locked by Spotify): {e}")
            
    # 2. CrashDumps (0.60 GB)
    crashdumps = r"C:\Users\admin\AppData\Local\CrashDumps"
    if os.path.exists(crashdumps):
        print("Cleaning System CrashDumps (.dmp files)...")
        files_deleted = 0
        for file in os.listdir(crashdumps):
            if file.endswith(".dmp"):
                filepath = os.path.join(crashdumps, file)
                try:
                    os.remove(filepath)
                    files_deleted += 1
                except Exception as e:
                    print(f"Failed to remove {file}: {e}")
        print(f"Deleted {files_deleted} crash dump files.")
        
    # 3. Temp Directory (1.44 GB)
    temp_path = r"C:\Users\admin\AppData\Local\Temp"
    if os.path.exists(temp_path):
        print("Cleaning C drive Windows Temp directory...")
        files_deleted = 0
        folders_deleted = 0
        for entry in os.scandir(temp_path):
            try:
                if entry.is_file(follow_symlinks=False):
                    os.remove(entry.path)
                    files_deleted += 1
                elif entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path)
                    folders_deleted += 1
            except Exception:
                # Skip locked files
                continue
        print(f"Deleted {files_deleted} files and {folders_deleted} directories in Temp.")
        
    print("\nCleanup finished!")

if __name__ == "__main__":
    main()
