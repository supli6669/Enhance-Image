# Use official lightweight Python image
FROM python:3.11-slim

# Install system dependencies required for OpenCV, Git and compiling packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set environment variables for non-interactive python behavior and logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Upgrade pip and install wheel/setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Optimization: Pre-install CPU-only PyTorch, Torchvision and NumPy to keep Docker image small
RUN pip install --no-cache-dir torch torchvision numpy --index-url https://download.pytorch.org/whl/cpu
# Copy tools directory (including patch script and base64 wheel) and install basicsr offline on Python 3.11+
COPY tools/ /app/tools/
RUN python tools/patch_and_install_basicsr.py
# Copy requirements and install the remaining packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including models/CodeFormer)
COPY . /app/

# Create weights directory and make sure it has permissions
RUN mkdir -p /app/weights && chmod -R 777 /app/weights

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run Streamlit on port 7860 in headless mode with CORS and XSRF disabled
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
