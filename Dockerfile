# Base image with PyTorch + CUDA (adjust if you're on CPU only)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Optional: switch to CPU if you don't need CUDA
# FROM pytorch/pytorch:2.1.0-cpu

# Set environment
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy your code
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip

# Install requirements
RUN pip install \
    numpy \
    scipy \
    scikit-image \
    matplotlib \
    ipykernel \
    nibabel \
    opencv-python-headless \
    torchmetrics \
    vedo \
    napari \
    jupyterlab

# Optional: install jupyter extensions or other dev tools

# Expose Jupyter port (optional)
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
