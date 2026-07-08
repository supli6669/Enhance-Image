---
title: Custom AI Face Enhancer
emoji: ✨
colorFrom: purple
colorTo: pink
sdk: streamlit
app_file: app.py
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
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Setup
1. Clone this repository to your local machine.
2. Initialize the virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # On Windows (PowerShell/CMD)
   source .venv/bin/activate   # On Linux/macOS
   ```
3. Run the custom BasicSR package patching script (required for Python 3.12+):
   ```bash
   python patch_and_install_basicsr.py
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

To deploy this app on **Hugging Face Spaces** for free:
1. Log in to [Hugging Face](https://huggingface.co/) and create a new **Space**.
2. Set the **SDK** to **Streamlit**.
3. Choose the **Free CPU Basic** tier (includes 16GB RAM, which is sufficient for running PyTorch CPU inference).
4. Connect this GitHub repository directly to the Space, or push the repository files to the Hugging Face Git remote.
5. Hugging Face will read the metadata frontmatter in this `README.md` file, install the libraries listed in `requirements.txt`, and automatically start the app. The pretrained weights will be downloaded programmatically at startup.
