"""
RunPod Serverless Handler — llama-cpp-python with Qwen3.5-122B-A10B GGUF

Loads the Qwen3.5-122B-A10B GGUF model at startup and exposes a REST-style
handler via RunPod's serverless infrastructure. No Flask, ngrok, or SQS needed.

Environment variables:
    HF_TOKEN            — HuggingFace auth token (optional, for gated models)
    MODEL_REPO_ID       — HF repo ID (default: unsloth/Qwen3.5-122B-A10B-GGUF)
    MODEL_QUANT_DIR     — Quant subdirectory (default: UD-Q4_K_XL)
    MODEL_FIRST_SPLIT   — First split filename (default: auto-detected)
    N_CTX               — Context window size (default: 16384)
    N_GPU_LAYERS        — GPU layers to offload, -1 for all (default: -1)
    NETWORK_VOLUME_PATH — Path to check for pre-cached models (default: /runpod-volume/models)
"""

import os
import re
import glob
import runpod
import traceback
from huggingface_hub import login, snapshot_download
from llama_cpp import Llama

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "unsloth/Qwen3.5-122B-A10B-GGUF")
MODEL_QUANT_DIR = os.environ.get("MODEL_QUANT_DIR", "UD-Q4_K_XL")
MODEL_FIRST_SPLIT = os.environ.get("MODEL_FIRST_SPLIT", "")
N_CTX = int(os.environ.get("N_CTX", "16384"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))
NETWORK_VOLUME_PATH = os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume/models")

# ============================================================================
# AUTHENTICATION
# ============================================================================

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("HuggingFace authentication successful!")
else:
    print("Warning: No HF_TOKEN found, proceeding without authentication")

# ============================================================================
# MODEL LOADING
# ============================================================================


def find_first_split(model_dir, quant_dir):
    """Find the first GGUF split file in the model directory."""
    quant_path = os.path.join(model_dir, quant_dir)
    if not os.path.isdir(quant_path):
        # Maybe files are directly in model_dir
        quant_path = model_dir

    # Look for split pattern (00001-of-NNNNN) or single file
    split_pattern = os.path.join(quant_path, "*-00001-of-*.gguf")
    splits = sorted(glob.glob(split_pattern))
    if splits:
        return splits[0]

    # Fallback: any .gguf file
    all_gguf = sorted(glob.glob(os.path.join(quant_path, "*.gguf")))
    if all_gguf:
        return all_gguf[0]

    raise FileNotFoundError(
        f"No GGUF files found in {quant_path}. "
        f"Contents: {os.listdir(quant_path) if os.path.isdir(quant_path) else 'dir not found'}"
    )


def load_model():
    """Download (or locate cached) GGUF model and initialize Llama."""

    # 1. Check network volume for pre-cached model
    repo_name = MODEL_REPO_ID.split("/")[-1]
    volume_model_path = os.path.join(NETWORK_VOLUME_PATH, repo_name, MODEL_QUANT_DIR)

    if os.path.isdir(volume_model_path):
        gguf_files = glob.glob(os.path.join(volume_model_path, "*.gguf"))
        if gguf_files:
            print(f"Found pre-cached model in network volume: {volume_model_path}")
            model_dir = os.path.join(NETWORK_VOLUME_PATH, repo_name)
            first_split = (
                os.path.join(volume_model_path, MODEL_FIRST_SPLIT)
                if MODEL_FIRST_SPLIT
                else find_first_split(model_dir, MODEL_QUANT_DIR)
            )
            print(f"Using model file: {first_split}")
            return Llama(
                model_path=first_split,
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
            )

    # 2. Download from HuggingFace
    print(f"Downloading model: {MODEL_REPO_ID} ({MODEL_QUANT_DIR})...")
    print("This may take a while for large models. Consider using a network volume.")
    model_dir = snapshot_download(
        repo_id=MODEL_REPO_ID,
        allow_patterns=[f"{MODEL_QUANT_DIR}/*"],
    )
    print(f"Model downloaded to: {model_dir}")

    # 3. Find the first split
    if MODEL_FIRST_SPLIT:
        first_split = os.path.join(model_dir, MODEL_QUANT_DIR, MODEL_FIRST_SPLIT)
    else:
        first_split = find_first_split(model_dir, MODEL_QUANT_DIR)

    print(f"Using model file: {first_split}")

    # 4. Initialize Llama
    return Llama(
        model_path=first_split,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
    )


print("=" * 60)
print(f"Loading model: {MODEL_REPO_ID} ({MODEL_QUANT_DIR})")
print(f"Context window: {N_CTX}, GPU layers: {N_GPU_LAYERS}")
print("=" * 60)

llm = load_model()
print("Model loaded successfully!")


# ============================================================================
# UTILITIES
# ============================================================================


def trim_thinking(text):
    """Remove <think>...</think> blocks from the model output.

    Qwen3.5 wraps its internal reasoning in <think>...</think> tags.
    For most use cases, only the final answer after </think> is needed.
    """
    # Try splitting on </think> first (handles unclosed tags too)
    if "</think>" in text:
        result = text.split("</think>")[-1].strip()
        if result:
            return result

    # Fallback: regex removal of complete think blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


# ============================================================================
# HANDLER
# ============================================================================


def handler(event):
    """RunPod serverless handler for llama-cpp-python chat completions.

    Input fields (in event["input"]):
        text_prompt (str, required):    The user's text prompt.
        image_base64 (str, optional):   Base64-encoded image data (no data URI prefix).
        image_media_type (str, optional): MIME type of the image. Default: image/jpeg.
        system_prompt (str, optional):  System prompt. Default: helpful assistant.
        chat_history (list, optional):  Prior messages [{role, content}, ...].
        max_tokens (int, optional):     Max tokens to generate. Default: 8000.
        temperature (float, optional):  Sampling temperature. Default: 0.6.
        top_p (float, optional):        Top-p sampling. Default: 0.95.
        top_k (int, optional):          Top-k sampling. Default: 20.
        min_p (float, optional):        Min-p sampling. Default: 0.0.
        enable_thinking (bool, optional): Keep <think> blocks. Default: False.

    Returns:
        dict with status, response (trimmed), and raw_response (full).
    """
    print("Worker received request")

    input_data = event.get("input", {})

    # --- Extract parameters ---
    text_prompt = input_data.get("text_prompt", "")
    image_base64 = input_data.get("image_base64")
    image_media_type = input_data.get("image_media_type", "image/jpeg")
    system_prompt = input_data.get("system_prompt", "You are a helpful assistant.")
    chat_history = input_data.get("chat_history", [])
    max_tokens = input_data.get("max_tokens", 8000)
    temperature = input_data.get("temperature", 0.6)
    top_p = input_data.get("top_p", 0.95)
    top_k = input_data.get("top_k", 20)
    min_p = input_data.get("min_p", 0.0)
    enable_thinking = input_data.get("enable_thinking", False)

    # --- Validate ---
    if not text_prompt:
        return {"status": "error", "error": "text_prompt is required"}

    # --- Build messages ---
    messages = []

    # System prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Chat history
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant", "system") and content:
                messages.append({"role": role, "content": content})

    # Current user message (with optional multimodal content)
    user_content = []
    if image_base64:
        data_uri = f"data:{image_media_type};base64,{image_base64}"
        user_content.append({
            "type": "image_url",
            "image_url": {"url": data_uri},
        })
    user_content.append({"type": "text", "text": text_prompt})

    messages.append({"role": "user", "content": user_content})

    print(f"Processing: {len(messages)} messages, max_tokens={max_tokens}")
    print(
        f"Params: temp={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}"
    )

    try:
        # --- Generate ---
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

        raw_response = response["choices"][0]["message"]["content"]

        # Trim thinking blocks unless caller wants them
        if enable_thinking:
            final_response = raw_response
        else:
            final_response = trim_thinking(raw_response)

        print(f"Generated {len(raw_response)} chars (trimmed: {len(final_response)})")

        return {
            "status": "success",
            "response": final_response,
            "raw_response": raw_response,
            "usage": response.get("usage", {}),
        }

    except Exception as e:
        print(f"Error generating completion: {e}")
        print(traceback.format_exc())
        return {"status": "error", "error": str(e)}


# ============================================================================
# ENTRYPOINT
# ============================================================================

# if __name__ == "__main__":
#     print("=" * 60)
#     print(f"Handler ready! Model: {MODEL_REPO_ID}")
#     print("RunPod serverless endpoint — no Flask/ngrok/SQS needed")
#     print("=" * 60)
runpod.serverless.start({"handler": handler})
