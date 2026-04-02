# RunPod Serverless Worker — llama-cpp-python with CUDA
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Build dependencies for llama-cpp-python CUDA compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /

# Install Python dependencies
# llama-cpp-python is compiled with CUDA support via CMAKE_ARGS
RUN python -m pip install --upgrade pip && \
    CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python && \
    pip install -r /requirements.txt

# Copy handler
COPY rp_handler.py /

WORKDIR /

# RunPod handler entrypoint
CMD ["python", "-u", "rp_handler.py"]
