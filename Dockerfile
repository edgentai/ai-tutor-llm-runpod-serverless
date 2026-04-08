# # RunPod Serverless Worker — llama-cpp-python with CUDA
# FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# # Build dependencies for llama-cpp-python CUDA compilation
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     cmake \
#     curl \
#     git \
#     libcurl4-openssl-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements
# COPY requirements.txt /

# # Install Python dependencies
# # CUDA driver stubs are needed at link time (the real driver is available at runtime on the GPU host)
# # Setting LIBRARY_PATH to cuda stubs resolves the "libcuda.so.1 not found" linker error
# ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

# RUN python -m pip install --upgrade pip && \
#     CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80;86;89;90" \
#     LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH} \
#     LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
#     pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python && \
#     pip install -r /requirements.txt


# # Copy handler
# COPY rp_handler.py /

# WORKDIR /

# # RunPod handler entrypoint
# CMD ["python", "-u", "rp_handler.py"]
# Use RunPod's Python 3.10 CUDA image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04


# Copy requirements file
COPY requirements.txt /

# Install dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /requirements.txt

# Copy your handler file
COPY rp_handler.py /

# Set the working directory
WORKDIR /

# RunPod handler command
CMD ["python", "-u", "rp_handler.py"]