---
title: Custom AI Face Enhancer
emoji: ✨
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: apache-2.0
---

# Custom AI Face Enhancer & Restorer

A self-hosted, local AI image enhancement web application built using **Python, Streamlit, PyTorch, and OpenCV**. It integrates the state-of-the-art **CodeFormer** model locally for face restoration with custom blending controls.

This project is configured to run out-of-the-box both **locally** on your CPU/GPU and **hosted on Hugging Face Spaces**.

---

## Key Features

- **Local Execution:** Runs directly on local CPU or NVIDIA GPU (via CUDA) for maximum privacy and processing speed.
- **Fidelity Weight Tuning ($w$):** Control the balance between generating rich realistic details (low $w$) and keeping high resemblance to the original face (high $w$).
- **Custom Blending Softness:** Exposes an adjustable soft mask feathering parameter to ensure smooth, seamless pasting of restored faces back into the upscaled background image.
- **Multiple Face Detectors:** Choose between highly accurate detectors (RetinaFace) or faster detectors for groups (YOLOv5).
- **Dark Mode UI:** Designed with custom glassmorphism and modern Outfit typography.

---

## Local Execution Instructions

### Prerequisites
- Python 3.11
- Git

### Setup
1. Clone this repository to your local machine.
2. Initialize the virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # On Windows (PowerShell/CMD)
   source .venv/bin/activate   # On Linux/macOS
   ```
3. Run the custom BasicSR package patching script:
   ```bash
   python tools/patch_and_install_basicsr.py
   ```
4. Install the remaining requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Start the Streamlit server locally:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The app will automatically download the pretrained model weights on its first run.

---

## Hugging Face Spaces Deployment

This repository deploys as a **Docker Space**, not a Streamlit SDK Space. The
Dockerfile installs the pinned Python 3.11 CPU runtime and starts Streamlit on
port `7860`.

1. Log in to [Hugging Face](https://huggingface.co/) and create a new Space.
2. Set the Space SDK to **Docker** and select a CPU hardware tier appropriate
   for CodeFormer inference.
3. Push this repository to the Space. Keep Git LFS enabled: the tracked
   `weights/CodeFormer/codeformer.pth` model is required at build/runtime.
4. Wait for the Space build to complete, then open the Space URL. Additional
   optional model files are downloaded by the application only if unavailable.

### GitHub Actions sync

The included workflow syncs `main` to the configured Hugging Face Space and
uploads Git LFS objects first. Add a Hugging Face **write** token as the GitHub
Actions secret `HF_TOKEN`; do not place a token in a Git remote URL or commit it
to the repository. The sync intentionally does not force-push, so resolve any
divergent Space changes before running it again.
