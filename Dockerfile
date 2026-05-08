# RunPod's PyTorch image with CUDA 13 — required for vLLM 0.19+, which the
# Qwen3.6 docs recommend for the hybrid Gated DeltaNet + Gated Attention
# stack. The previous CUDA-12.8 base could not load vLLM 0.19 wheels and
# crashed at import with `libcudart.so.13: cannot open shared object file`.
#
# Tag scheme note: RunPod only ships CUDA 13 on Ubuntu 24.04 (no 22.04
# variant exists), so Python is 3.12 here, not 3.11. vLLM 0.19 ships
# cp38-abi3 stable-ABI wheels, which load on Python 3.8+ — verified against
# the PyPI wheel index. If you need to pin to a different PyTorch line, the
# only other CUDA-13 option is `1.0.3-cu1300-torch291-ubuntu2404`.
FROM runpod/pytorch:1.0.3-cu1300-torch290-ubuntu2404

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

# vLLM 0.19.0 — recommended minimum per the Qwen3.6 model docs for the
# hybrid 16 × (3 × Gated DeltaNet + 1 × Gated Attention) stack. Built
# against CUDA 13, which is why the base image was bumped above. Pin exact:
# vLLM silently rolls model-config defaults between minors.
RUN python -m pip install --upgrade pip && \
    pip install vllm==0.19.0 && \
    pip install -r /requirements.txt

# Handler + model wrapper
COPY rp_handler.py /
COPY models /models

WORKDIR /

CMD ["python", "-u", "rp_handler.py"]
