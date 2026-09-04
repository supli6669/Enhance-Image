import json

p = 'train_kaggle.ipynb'
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)

data['cells'][3]['source'] = [
    "# 4. Launch GPU Training with ArcFace Identity Loss\n",
    "%cd /kaggle/working/custom-ai-enhancer\n",
    "import os, glob, shutil\n",
    "os.environ[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"\n",
    "\n",
    "# Check for previous checkpoints from attached kernel data source\n",
    "found_ckpts = glob.glob('/kaggle/input/**/models/net_g_*.pth', recursive=True)\n",
    "if found_ckpts:\n",
    "    print(f\"[Resume] Found {len(found_ckpts)} checkpoint(s) in /kaggle/input/. Setting up resume state...\")\n",
    "    target_exp_dir = 'models/CodeFormer/experiments/20260904_114846_CodeFormer_stage3_custom'\n",
    "    os.makedirs(os.path.join(target_exp_dir, 'models'), exist_ok=True)\n",
    "    os.makedirs(os.path.join(target_exp_dir, 'training_states'), exist_ok=True)\n",
    "    for ckpt in glob.glob('/kaggle/input/**/models/*.pth', recursive=True):\n",
    "        shutil.copy(ckpt, os.path.join(target_exp_dir, 'models', os.path.basename(ckpt)))\n",
    "    for st_f in glob.glob('/kaggle/input/**/training_states/*.state', recursive=True):\n",
    "        shutil.copy(st_f, os.path.join(target_exp_dir, 'training_states', os.path.basename(st_f)))\n",
    "    print(\"[Resume] Previous experiment state restored successfully!\")\n",
    "\n",
    "!python train_custom.py\n"
]

with open(p, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

print("train_kaggle.ipynb updated successfully!")
