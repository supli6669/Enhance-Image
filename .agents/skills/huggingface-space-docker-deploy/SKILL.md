---
name: huggingface-space-docker-deploy
description: Packaging AI web apps into lightweight Docker containers for HuggingFace Spaces deployment, managing CPU thread limits, memory leaks, and model downloading caches.
---

# HuggingFace Space Docker Deployment & Optimization Skill

## Overview
Deploying AI web applications to HuggingFace Spaces via Docker SDK allows full environment control, CPU PyTorch optimization, custom system packages (`libgl1-mesa-glx`), and automated runtime weight downloading.

## Standard Dockerfile Template
```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
```
