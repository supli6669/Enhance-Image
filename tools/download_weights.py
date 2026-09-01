import os
import urllib.request
import sys

def download_file(url, save_path):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        print(f"File already exists: {save_path}. Skipping.")
        return
        
    print(f"Downloading {url} to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response, open(save_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024 # 1 MB chunks
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    pct = downloaded * 100.0 / total_size
                    sys.stdout.write(f"\rProgress: {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB of {total_size / 1024 / 1024:.1f} MB)")
                    sys.stdout.flush()
        print("\nDownload finished successfully.")
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass
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
