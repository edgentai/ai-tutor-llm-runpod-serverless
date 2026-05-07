# RunPod's PyTorch image with CUDA support
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# CUDA env
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# vLLM V1's AsyncLLMEngine spawns an EngineCore subprocess. Force `spawn`
# instead of fork — fork dies with "Cannot re-initialize CUDA" when CUDA
# state already exists in the parent (which it does after NIXL probes /
# model-config loading). Slightly slower startup, but actually works.
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Quieter per-request logs under load. Flip to INFO for debugging.
ENV VLLM_LOGGING_LEVEL=WARNING

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
COPY rp_handler.py /
COPY models /models

WORKDIR /

CMD ["python", "-u", "rp_handler.py"]
