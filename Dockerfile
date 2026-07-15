# Use official lightweight Python image
FROM python:3.11-slim

# Install system dependencies required for OpenCV, Git and compiling packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install wheel/setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Optimization: Pre-install CPU-only PyTorch & Torchvision to keep Docker image small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and execute the custom BasicSR patch script to install basicsr on Python 3.11+
COPY patch_and_install_basicsr.py /app/
RUN python patch_and_install_basicsr.py

# Copy requirements and install the remaining packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including models/CodeFormer)
COPY . /app/

# Create weights directory and make sure it has permissions
RUN mkdir -p /app/weights && chmod -R 777 /app/weights

# Download model weights during build phase to prevent runtime startup hang
RUN python tools/download_weights.py

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run Streamlit on port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
