"""RunPod serverless handler — single-tenant LLM service (Qwen3.6-27B).

Async by design: vLLM's continuous batching only kicks in if the worker can
have multiple in-flight requests at once. We:
  1. Use AsyncLLMEngine in models.llm
  2. Make `handler` an `async def`
  3. Tell RunPod to send up to LLM_MAX_SEQS concurrent jobs to one worker
     via the `concurrency_modifier` callback below.

So one A100-80GB worker batches up to 32 simultaneous chats into single
forward passes, and 100 concurrent users see ~25 s p50 / ~50 s p99 instead
of serialising at ~30 min behind a single stream.

INPUT (event["input"]):
  Either:
    messages: [{"role":..., "content":...}, ...]  (preferred)
  Or:
    text_prompt: "..."                            (flat shorthand)

  Optional:
    system_prompt:    str
    image_source:     str   (http(s) URL or "data:image/...;base64,...")
    max_tokens:       int   (default 2048)
    temperature:      float
    top_p:            float
    top_k:            int
    enable_thinking:  bool  (default True)
    presence_penalty: float
    return_thinking:  bool  (default False)

OUTPUT:
  {status, response: {choices, model, usage}}  (OpenAI-compatible)
"""
import os
import traceback
from typing import Any, Dict

import runpod
from huggingface_hub import login

from models import llm as llm_mod

# ============================================================================
# AUTH
# ============================================================================
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("HuggingFace authentication successful.")
else:
    print("Warning: No HF_TOKEN found, proceeding without authentication")


# ============================================================================
# MODEL INIT — at import time, so the worker is ready by the first request
# ============================================================================
llm_mod.initialize()


# ============================================================================
# HANDLER
# ============================================================================
async def handle_llm(input_data: Dict[str, Any]) -> Dict[str, Any]:
    if not llm_mod.is_ready():
        return {"status": "error", "error": "LLM not initialized on this worker."}

    # Accept either `messages` (preferred) or flat `text_prompt`.
    messages         = input_data.get("messages")
    text_prompt      = input_data.get("text_prompt")
    system_prompt    = input_data.get("system_prompt", "")
    image_source     = input_data.get("image_source")
    max_tokens       = input_data.get("max_tokens", 2048)
    temperature      = input_data.get("temperature")
    top_p            = input_data.get("top_p")
    top_k            = input_data.get("top_k")
    enable_thinking  = input_data.get("enable_thinking", True)
    presence_penalty = input_data.get("presence_penalty")
    return_thinking  = input_data.get("return_thinking", False)

    if not messages and text_prompt:
        messages = [{"role": "user", "content": text_prompt}]
    if not messages:
        return {
            "status": "error",
            "error":  "Either `messages` or `text_prompt` is required.",
        }

    chat_messages = llm_mod.build_chat_messages(
        system_prompt=system_prompt,
        user_messages=messages,
        image_source=image_source,
    )
    print(f"  messages={len(chat_messages)}  thinking={enable_thinking}  "
          f"max_tokens={max_tokens}  image={'yes' if image_source else 'no'}")

    try:
        result = await llm_mod.chat_generate(
            chat_messages,
            thinking_mode=enable_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            return_thinking=return_thinking,
        )

        message: Dict[str, Any] = {
            "role":    "assistant",
            "content": result["response"],
        }
        if return_thinking and result["thinking"]:
            message["thinking"] = result["thinking"]

        response = {
            "choices": [{
                "message":       message,
                "finish_reason": result["finish_reason"],
                "index":         0,
            }],
            "model": llm_mod.MODEL_ID,
            "usage": {
                "prompt_tokens":     result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens":      result["prompt_tokens"] + result["completion_tokens"],
            },
            "request_id": result["request_id"],
        }
        return {"status": "success", "response": response}

    except Exception as e:
        print(f"Error in LLM handler: {e}")
        print(traceback.format_exc())
        return {"status": "error", "error": str(e)}


async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Single-purpose LLM handler. `feature_flag` is accepted but ignored."""
    return await handle_llm(event.get("input", {}))


# ============================================================================
# CONCURRENCY MODIFIER
# Tells RunPod how many concurrent jobs this worker is willing to take. We
# match LLM_MAX_SEQS so vLLM's continuous batcher always has work for every
# slot it can schedule, but never more than it can hold KV cache for.
#
# RunPod calls this periodically with the current concurrency level; we
# return the *target* level. Static = always allow MAX_NUM_SEQS in flight.
# ============================================================================
def adjust_concurrency(_current_concurrency: int) -> int:
    return llm_mod.MAX_NUM_SEQS


# ============================================================================
# ENTRYPOINT
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"LLM service ready | model={llm_mod.MODEL_ID} | concurrency={llm_mod.MAX_NUM_SEQS}")
    print("=" * 60)
    runpod.serverless.start({
        "handler":              handler,
        "concurrency_modifier": adjust_concurrency,
    })
