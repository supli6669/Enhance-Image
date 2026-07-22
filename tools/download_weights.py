import os
import urllib.request
import sys

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}. Skipping.")
        return
        
    print(f"Downloading {url} to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Simple progress reporter
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, read_so_far * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent:.2f}% ({read_so_far / 1024 / 1024:.2f} MB of {total_size / 1024 / 1024:.2f} MB)")
        else:
            sys.stdout.write(f"\rProgress: {read_so_far / 1024 / 1024:.2f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, save_path, reporthook=report_hook)
        print("\nDownload finished successfully.")
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        raise e

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    weights_to_download = {
        "weights/CodeFormer/codeformer.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "weights/facelib/detection_Resnet50_Final.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "weights/facelib/detection_mobilenet0.25_Final.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth",
        "weights/facelib/parsing_parsenet.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
        "weights/facelib/yolov5l-face.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/yolov5l-face.pth",
        "weights/facelib/vqgan_code1024.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/vqgan_code1024.pth",
        "weights/realesrgan/RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"


    }
    
    for rel_path, url in weights_to_download.items():
        save_path = os.path.join(project_dir, rel_path)
        try:
            download_file(url, save_path)
        except Exception as e:
            print(f"Failed to download weight: {save_path}")
            sys.exit(1)
            
    print("All model weights have been downloaded and placed in the weights/ folder.")

if __name__ == "__main__":
    main()
