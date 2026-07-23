# Pinned Debian/Python base for reproducible CPU builds.
FROM python:3.11.9-slim-bookworm

# Install system dependencies required for OpenCV, Git and compiling packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set environment variables for non-interactive python behavior, logging, and writable cache paths
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    TORCH_HOME=/tmp \
    HF_HOME=/tmp

# Upgrade packaging tooling to a known version.
RUN pip install --no-cache-dir --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0

# Install the tested CPU-only PyTorch pair before the remaining application dependencies.
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the remaining packages before BasicSR so its
# installer resolves against the pinned versions below.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy tools directory (including patch script and base64 wheel) and install basicsr offline on Python 3.11+
COPY tools/ /app/tools/
RUN python tools/patch_and_install_basicsr.py

# Copy all project files (including models/CodeFormer)
COPY . /app/

# Create weights directories and make sure all permissions are wide open for Hugging Face non-root user (uid 1000)
RUN mkdir -p /app/weights /app/weights/CodeFormer /app/weights/realesrgan /app/weights/facelib /tmp \
    && chmod -R 777 /app /tmp

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run Streamlit with its secure CORS and XSRF defaults enabled.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
