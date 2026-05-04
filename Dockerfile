# RunPod's PyTorch image with CUDA support
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# vLLM cache + CUDA env
ENV VLLM_USE_CACHE=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Build deps only — no ffmpeg, no flash-attn (vLLM ships its own attention).
RUN apt-get update && apt-get install -y \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /

# vLLM nightly (Qwen3.x hybrid arch support) + minimal python deps.
RUN python -m pip install --upgrade pip && \
    pip install vllm --extra-index-url https://wheels.vllm.ai/nightly && \
    pip install -r /requirements.txt

# Handler + model wrapper
COPY handler.py /
COPY models /models

WORKDIR /

CMD ["python", "-u", "rp_handler.py"]
