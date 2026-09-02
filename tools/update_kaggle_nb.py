import json

p = 'train_kaggle.ipynb'
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)

data['cells'][1]['source'] = [
    "import os, sys, subprocess, shutil, torch\n",
    "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU Device: {torch.cuda.get_device_name(0)}\")\n",
    "    try:\n",
    "        c = torch.nn.Conv2d(3, 3, 3).cuda()\n",
    "        x = torch.randn(1, 3, 32, 32, device='cuda')\n",
    "        _ = c(x)\n",
    "        print(\"Native CUDA Conv2D verification PASSED!\")\n",
    "    except Exception as e:\n",
    "        print(f\"CUDA check failed ({e}). Installing PyTorch with Tesla P100 (sm_60) support...\")\n",
    "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.4.0+cu121', 'torchvision==0.19.0+cu121', '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)\n",
    "        print('Compatible PyTorch installed!')\n",
    "else:\n",
    "    print('WARNING: Running on CPU!')\n"
]

with open(p, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

print("train_kaggle.ipynb updated successfully!")
