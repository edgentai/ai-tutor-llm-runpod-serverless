"""Qwen3.6-27B AsyncLLMEngine wrapper — tuned for A100-80GB single tenant.

Why async, not sync `LLM`:
  Sync `vllm.LLM.chat()` blocks the worker thread end-to-end, so vLLM only
  ever sees one in-flight sequence per RunPod worker — `max_num_seqs=32`
  pays for nothing, continuous batching is wasted, and 100 concurrent users
  serialise behind a single stream.

  AsyncLLMEngine + an async handler + RunPod concurrency_modifier lets
  vLLM batch up to MAX_NUM_SEQS simultaneous chats into one forward pass
  with continuous batching. This is the *only* configuration that
  actually uses the optimisations below.

Optimisations applied:
  1. Sampling tuned per mode (Qwen3 docs): thinking → T=0.6, P=0.95, K=20;
     non-thinking → T=0.7, P=0.8, K=20. Never greedy.
  2. YaRN long-context: native 32K → 128K via rope_scaling, gated on
     max_model_len > native ctx (vLLM warns/fails otherwise).
  3. Prefix caching ON — 2-5x speedup on reused system prompts.
  4. CUDA graphs ON (enforce_eager=False default) — 10-20% faster decode on
     A100. One-time ~30-60s graph capture cost paid at startup.
  5. n-gram speculative decoding — free 1.5-2x decode throughput on RAG /
     structured-output workloads. No draft model, no extra GPU memory.
  6. enable_chunked_prefill explicitly ON — long-ctx prefills don't block
     decode batches. Important when MAX_MODEL_LEN is 128K.
  7. gpu_memory_utilization=0.92 — squeezes ~3 GB more KV cache out of the
     A100-80GB after subtracting the bf16 27B weights (~54 GB).
  8. Multimodal: vLLM auto-detects vision; `image_source` / image_url parts
     in messages are downloaded and passed via multi_modal_data.
  9. <think>…</think> stripper handles both `…</think>final` (chat template
     ate the opening tag — usual case) and `<think>…</think>final`.

Knobs:
  LLM_ENFORCE_EAGER=1 → skip CUDA graph compile (fast cold start, slower
                       steady state — pick this only if traffic is sporadic)
  LLM_SPECULATIVE=0   → disable n-gram speculation (default ON)
  LLM_GPU_MEM=0.88    → if you need to leave headroom for another tenant
"""
import asyncio
import base64
import os
import re
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

# ---------------------------------------------------------------------------
# CONFIG — env-var overridable
# ---------------------------------------------------------------------------
MODEL_ID                 = os.getenv("LLM_MODEL_ID", "Qwen/Qwen3.6-27B")
TENSOR_PARALLEL_SIZE     = int(os.getenv("LLM_TP", "1"))
GPU_MEM_UTIL             = float(os.getenv("LLM_GPU_MEM", "0.92"))
MAX_MODEL_LEN            = int(os.getenv("LLM_MAX_MODEL_LEN", "131072"))
NATIVE_CTX               = int(os.getenv("LLM_NATIVE_CTX", "32768"))
YARN_FACTOR              = float(os.getenv("LLM_YARN_FACTOR", "4.0"))
MAX_NUM_SEQS             = int(os.getenv("LLM_MAX_SEQS", "32"))
# vLLM warns when max_num_batched_tokens is too small for spec-decode draft
# slots: with num_spec_tokens=5 and max_num_seqs=32, the prefill+draft budget
# wants more headroom than the 2048 default. 8192 silences the warning and
# lets prefill batches actually use the budget.
MAX_NUM_BATCHED_TOKENS   = int(os.getenv("LLM_MAX_BATCHED_TOKENS", "8192"))
ENABLE_PREFIX_CACHE      = os.getenv("LLM_PREFIX_CACHE", "1") == "1"
ENABLE_CHUNKED_PREFILL   = os.getenv("LLM_CHUNKED_PREFILL", "1") == "1"
LIMIT_IMAGES_PER_PROMPT  = int(os.getenv("LLM_IMAGES_PER_PROMPT", "4"))
ENFORCE_EAGER            = os.getenv("LLM_ENFORCE_EAGER", "0") == "1"
WARMUP_ON_LOAD           = os.getenv("LLM_WARMUP", "1") == "1"

SPECULATIVE_ENABLED      = os.getenv("LLM_SPECULATIVE", "1") == "1"
SPECULATIVE_NUM_TOKENS   = int(os.getenv("LLM_SPEC_NUM_TOKENS", "5"))
SPECULATIVE_LOOKUP_MAX   = int(os.getenv("LLM_SPEC_LOOKUP_MAX", "4"))
SPECULATIVE_LOOKUP_MIN   = int(os.getenv("LLM_SPEC_LOOKUP_MIN", "2"))

IMAGE_DOWNLOAD_TIMEOUT_S = int(os.getenv("LLM_IMG_TIMEOUT_S", "30"))

# Sampling presets per Qwen3 docs.
SAMPLING_THINKING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
SAMPLING_DIRECT   = dict(temperature=0.7, top_p=0.8,  top_k=20, min_p=0.0)
PRESENCE_PENALTY  = 1.0   # mild — Qwen3 docs: too high causes language mixing

# Strip pattern: greedily consume up to and including the first </think>.
# count=1 in re.sub ensures we only kill the leading reasoning block, never
# any legitimate </think>-shaped text deeper in the answer.
_THINK_PREFIX_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.DOTALL)

_engine: Optional[AsyncLLMEngine] = None
_tokenizer = None


# ---------------------------------------------------------------------------
# LIFECYCLE
# ---------------------------------------------------------------------------
def initialize() -> None:
    """Load Qwen3.6-27B into AsyncLLMEngine. Idempotent. Safe to call from
    sync module-init context."""
    global _engine, _tokenizer
    if _engine is not None:
        return

    print(f"[LLM] Loading {MODEL_ID} (AsyncLLMEngine)")
    print(f"[LLM]   max_model_len={MAX_MODEL_LEN}  YaRN factor={YARN_FACTOR}  native_ctx={NATIVE_CTX}")
    print(f"[LLM]   gpu_mem_util={GPU_MEM_UTIL}  prefix_cache={ENABLE_PREFIX_CACHE}  "
          f"chunked_prefill={ENABLE_CHUNKED_PREFILL}  max_seqs={MAX_NUM_SEQS}  "
          f"enforce_eager={ENFORCE_EAGER}  spec_decode={SPECULATIVE_ENABLED}")

    t0 = time.monotonic()

    # YaRN — only enable if the requested ctx exceeds native. vLLM 0.7+
    # moved this from a top-level kwarg into hf_overrides.
    hf_overrides: Dict[str, Any] = {}
    if MAX_MODEL_LEN > NATIVE_CTX:
        hf_overrides["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": YARN_FACTOR,
            "original_max_position_embeddings": NATIVE_CTX,
        }

    # n-gram speculation — vLLM proposes draft tokens by matching prompt-tail
    # n-grams against the running generation. Verified in one forward pass.
    speculative_config: Optional[Dict[str, Any]] = None
    if SPECULATIVE_ENABLED:
        speculative_config = {
            "method":                "ngram",
            "num_speculative_tokens": SPECULATIVE_NUM_TOKENS,
            "prompt_lookup_max":      SPECULATIVE_LOOKUP_MAX,
            "prompt_lookup_min":      SPECULATIVE_LOOKUP_MIN,
        }

    args_kwargs: Dict[str, Any] = dict(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=ENABLE_PREFIX_CACHE,
        enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
        hf_overrides=hf_overrides,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": LIMIT_IMAGES_PER_PROMPT},
        enforce_eager=ENFORCE_EAGER,
        disable_custom_all_reduce=True,
        # NB: `disable_log_requests` was removed from AsyncEngineArgs in
        # recent vLLM versions. Set VLLM_LOGGING_LEVEL=WARNING via env if
        # you want quieter per-request logs.
    )
    if speculative_config is not None:
        args_kwargs["speculative_config"] = speculative_config

    engine_args = AsyncEngineArgs(**args_kwargs)
    _engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"[LLM] Engine ready in {time.monotonic() - t0:.1f}s")

    if WARMUP_ON_LOAD:
        # Absorb CUDA graph capture / kernel autotune now so the first real
        # request isn't 10x slower than steady state. asyncio.run spins a
        # one-shot loop; runpod will create its own loop afterwards.
        print("[LLM] Warming up...")
        asyncio.run(_async_warmup())
        print("[LLM] Warmup complete.")


async def _async_warmup() -> None:
    """One short generation to trigger CUDA-graph capture + cache tokenizer."""
    global _tokenizer
    # get_tokenizer() is async in vLLM < 0.7, sync in vLLM >= 0.7
    result = _engine.get_tokenizer()
    _tokenizer = await result if asyncio.iscoroutine(result) else result

    # Use a chat-templated prompt — passing raw strings is deprecated since
    # vLLM 0.17 and will hard-fail in v0.18.
    warmup_prompt = _tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    sampling = SamplingParams(max_tokens=8, temperature=0.6, top_p=0.95)
    rid = "warmup-" + uuid.uuid4().hex[:8]
    async for _ in _engine.generate({"prompt": warmup_prompt}, sampling, request_id=rid):
        pass


def is_ready() -> bool:
    return _engine is not None


def get_engine() -> AsyncLLMEngine:
    if _engine is None:
        raise RuntimeError("LLM not initialised — call models.llm.initialize() first.")
    return _engine


async def get_tokenizer():
    """Lazy-load and cache the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        if _engine is None:
            raise RuntimeError("LLM not initialised.")
        result = _engine.get_tokenizer()
        _tokenizer = await result if asyncio.iscoroutine(result) else result
    return _tokenizer


# ---------------------------------------------------------------------------
# RESPONSE PROCESSING
# ---------------------------------------------------------------------------
def strip_thinking(text: str) -> str:
    """Remove the leading <think>…</think> reasoning block from output."""
    return _THINK_PREFIX_RE.sub("", text, count=1).strip()


def extract_thinking(text: str) -> str:
    """Return just the leading reasoning block (for return_thinking=True)."""
    m = _THINK_PREFIX_RE.match(text)
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------------------------
def build_sampling_params(
    *,
    max_tokens: int,
    thinking_mode: bool,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    repetition_penalty: float = 1.0,
) -> SamplingParams:
    base = dict(SAMPLING_THINKING if thinking_mode else SAMPLING_DIRECT)
    if temperature is not None:
        base["temperature"] = temperature
    if top_p is not None:
        base["top_p"] = top_p
    if top_k is not None:
        base["top_k"] = top_k
    return SamplingParams(
        **base,
        presence_penalty=PRESENCE_PENALTY if presence_penalty is None else presence_penalty,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# CHAT MESSAGE BUILDING + IMAGE LOADING
# ---------------------------------------------------------------------------
def build_chat_messages(
    system_prompt: Optional[str],
    user_messages: List[Dict[str, Any]],
    image_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Combine system prompt + user/assistant turns + optional image.

    If `image_source` is provided, it's attached to the LAST user message as
    a multimodal content list — matches the colab/llm.py contract.
    """
    chat: List[Dict[str, Any]] = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})

    for m in user_messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if role in ("user", "assistant", "system"):
            chat.append({"role": role, "content": content})

    if image_source and chat:
        for i in range(len(chat) - 1, -1, -1):
            if chat[i]["role"] == "user":
                text = chat[i]["content"] if isinstance(chat[i]["content"], str) else ""
                chat[i]["content"] = [
                    {"type": "image_url", "image_url": {"url": image_source}},
                    {"type": "text",      "text": text},
                ]
                break
    return chat


def _load_image(source: str):
    """Load a PIL Image from an http(s) URL or `data:image/...;base64,...` URI."""
    from PIL import Image  # imported lazily so the LLM-only code path doesn't pay for it

    if source.startswith("data:"):
        try:
            _header, b64 = source.split(",", 1)
        except ValueError:
            raise ValueError("Malformed data URI for image_source")
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    if source.startswith(("http://", "https://")):
        with urlopen(source, timeout=IMAGE_DOWNLOAD_TIMEOUT_S) as resp:
            return Image.open(resp).convert("RGB")
    raise ValueError(f"Unsupported image source scheme: {source[:32]}…")


def _extract_images(messages: List[Dict[str, Any]]) -> list:
    """Walk the chat messages and download any image_url parts to PIL Images."""
    images = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    if url:
                        images.append(_load_image(url))
    return images


# ---------------------------------------------------------------------------
# CHAT GENERATION (ASYNC)
# ---------------------------------------------------------------------------
async def chat_generate(
    chat_messages: List[Dict[str, Any]],
    *,
    thinking_mode: bool = True,
    max_tokens: int = 8192,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    repetition_penalty: float = 1.0,
    return_thinking: bool = False,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a chat completion using AsyncLLMEngine.

    Concurrent calls from different coroutines are merged by vLLM's scheduler
    into one forward pass (continuous batching), up to `max_num_seqs`.
    """
    if _engine is None:
        raise RuntimeError("LLM not initialised — call models.llm.initialize().")

    tokenizer = await get_tokenizer()
    sampling  = build_sampling_params(
        max_tokens=max_tokens,
        thinking_mode=thinking_mode,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )

    # Apply Qwen3 chat template. enable_thinking is consumed by the Jinja
    # template — extra kwargs to apply_chat_template flow into template vars.
    prompt_str = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking_mode,
    )

    images = _extract_images(chat_messages)

    engine_input: Dict[str, Any] = {"prompt": prompt_str}
    if images:
        engine_input["multi_modal_data"] = {"image": images}

    rid = request_id or uuid.uuid4().hex
    final_output = None
    async for request_output in _engine.generate(engine_input, sampling, rid):
        final_output = request_output

    if final_output is None or not final_output.outputs:
        raise RuntimeError("AsyncLLMEngine returned no outputs")

    raw           = final_output.outputs[0].text
    finish_reason = final_output.outputs[0].finish_reason or "unknown"
    cleaned       = strip_thinking(raw)
    thinking      = extract_thinking(raw) if return_thinking else None

    return {
        "response":          cleaned,
        "thinking":          thinking,
        "raw":               raw,
        "finish_reason":     finish_reason,
        "prompt_tokens":     len(final_output.prompt_token_ids or []),
        "completion_tokens": len(final_output.outputs[0].token_ids or []),
        "request_id":        rid,
    }
