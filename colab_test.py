"""
Colab end-to-end test for the Qwen3 vLLM serverless worker.
Full production config (Qwen/Qwen3.6-27B, A100-80GB).

Paste this entire file into a Colab cell (GPU runtime required),
then call:
    await run_all_tests()

Or run as a script:
    !python colab_test.py

Prerequisites:
    !pip install vllm transformers pillow nest_asyncio
    # Set HF_TOKEN if the model is gated:
    import os; os.environ["HF_TOKEN"] = "hf_..."
"""

# ── CRITICAL: set spawn BEFORE any import that could touch CUDA ───────────────
# vLLM V1 spawns an EngineCore subprocess. If CUDA is already initialised in
# the parent (e.g. by `import torch` + any cuda call) the subprocess hangs at
# spawn because CUDA state can't be transferred across the process boundary.
# Rule: nothing that touches CUDA may run before AsyncLLMEngine is constructed.
import multiprocessing as mp
import os
import subprocess
import sys

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Belt-and-braces: also set the vLLM-specific env var so worker processes
# spawned inside vLLM's own launcher also use spawn.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# ── GPU pre-flight via nvidia-smi (zero CUDA initialisation) ──────────────────
# Do NOT import torch or call any torch.cuda.* here — that would initialise
# CUDA in this process and cause the spawn deadlock described above.
try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if smi.returncode == 0:
        for line in smi.stdout.strip().splitlines():
            name, mib = line.split(",", 1)
            vram_gb = int(mib.strip()) / 1024
            print(f"[GPU] {name.strip()}  VRAM: {vram_gb:.1f} GB")
            if vram_gb < 75:
                print(
                    f"[GPU] WARNING — Qwen3.6-27B bf16 needs ~54 GB weights + KV cache.\n"
                    f"      Only {vram_gb:.1f} GB detected. Reduce MAX_MODEL_LEN or GPU_MEM_UTIL."
                )
    else:
        print("[GPU] nvidia-smi not available — cannot pre-check VRAM.")
except Exception as e:
    print(f"[GPU] pre-flight skipped: {e}")

# ── Protobuf compatibility shim ───────────────────────────────────────────────
# protobuf ≥ 4.x removed MessageFactory.GetPrototype; some vLLM deps still
# use the old API. Setting this env var forces the pure-Python implementation
# which doesn't have the issue. Must be set before any protobuf import.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# ── Standard library imports (no CUDA) ────────────────────────────────────────
import asyncio
import base64
import math
import re
import statistics
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional

import nest_asyncio
nest_asyncio.apply()

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from huggingface_hub import login as hf_login

# ── HuggingFace auth (set HF_TOKEN env var before running) ────────────────────
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    hf_login(token=_hf_token)
    print("[HF] Authenticated.")
else:
    print("[HF] No HF_TOKEN — proceeding unauthenticated (fine for public models).")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — exact mirror of models/llm.py production defaults
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID                 = os.getenv("LLM_MODEL_ID",             "Qwen/Qwen3.6-27B")
TENSOR_PARALLEL_SIZE     = int(os.getenv("LLM_TP",               "1"))
GPU_MEM_UTIL             = float(os.getenv("LLM_GPU_MEM",        "0.92"))
MAX_MODEL_LEN            = int(os.getenv("LLM_MAX_MODEL_LEN",    "131072"))
NATIVE_CTX               = int(os.getenv("LLM_NATIVE_CTX",       "32768"))
YARN_FACTOR              = float(os.getenv("LLM_YARN_FACTOR",    "4.0"))
MAX_NUM_SEQS             = int(os.getenv("LLM_MAX_SEQS",          "32"))
MAX_NUM_BATCHED_TOKENS   = int(os.getenv("LLM_MAX_BATCHED_TOKENS","8192"))
ENABLE_PREFIX_CACHE      = os.getenv("LLM_PREFIX_CACHE",   "1") == "1"
ENABLE_CHUNKED_PREFILL   = os.getenv("LLM_CHUNKED_PREFILL", "1") == "1"
LIMIT_IMAGES_PER_PROMPT  = int(os.getenv("LLM_IMAGES_PER_PROMPT","4"))
ENFORCE_EAGER            = os.getenv("LLM_ENFORCE_EAGER",   "0") == "1"
WARMUP_ON_LOAD           = os.getenv("LLM_WARMUP",          "1") == "1"
SPECULATIVE_ENABLED      = os.getenv("LLM_SPECULATIVE",     "1") == "1"
SPECULATIVE_NUM_TOKENS   = int(os.getenv("LLM_SPEC_NUM_TOKENS",  "5"))
SPECULATIVE_LOOKUP_MAX   = int(os.getenv("LLM_SPEC_LOOKUP_MAX",  "4"))
SPECULATIVE_LOOKUP_MIN   = int(os.getenv("LLM_SPEC_LOOKUP_MIN",  "2"))
IMAGE_DOWNLOAD_TIMEOUT_S = int(os.getenv("LLM_IMG_TIMEOUT_S",    "30"))

SAMPLING_THINKING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
SAMPLING_DIRECT   = dict(temperature=0.7, top_p=0.8,  top_k=20, min_p=0.0)
PRESENCE_PENALTY  = 1.0

_THINK_PREFIX_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.DOTALL)

_engine:    Optional[AsyncLLMEngine] = None
_tokenizer = None


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────
def initialize() -> None:
    global _engine, _tokenizer
    if _engine is not None:
        return

    print(f"\n[LLM] Loading {MODEL_ID}")
    print(f"[LLM]   max_model_len={MAX_MODEL_LEN}  YaRN factor={YARN_FACTOR}  native_ctx={NATIVE_CTX}")
    print(f"[LLM]   gpu_mem={GPU_MEM_UTIL}  prefix_cache={ENABLE_PREFIX_CACHE}  "
          f"chunked_prefill={ENABLE_CHUNKED_PREFILL}  max_seqs={MAX_NUM_SEQS}")
    print(f"[LLM]   enforce_eager={ENFORCE_EAGER}  spec_decode={SPECULATIVE_ENABLED}")
    print("[LLM]   (27B bf16 ~54 GB weights — model load takes ~5-10 min on first run)\n")
    t0 = time.monotonic()

    hf_overrides: Dict[str, Any] = {}
    if MAX_MODEL_LEN > NATIVE_CTX:
        hf_overrides["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": YARN_FACTOR,
            "original_max_position_embeddings": NATIVE_CTX,
        }

    speculative_config = None
    if SPECULATIVE_ENABLED:
        speculative_config = {
            "method":                 "ngram",
            "num_speculative_tokens":  SPECULATIVE_NUM_TOKENS,
            "prompt_lookup_max":       SPECULATIVE_LOOKUP_MAX,
            "prompt_lookup_min":       SPECULATIVE_LOOKUP_MIN,
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
    )
    if speculative_config is not None:
        args_kwargs["speculative_config"] = speculative_config

    engine_args = AsyncEngineArgs(**args_kwargs)
    _engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"[LLM] Engine ready in {time.monotonic() - t0:.1f}s")

    if WARMUP_ON_LOAD:
        print("[LLM] Warming up (CUDA graph capture) …")
        asyncio.run(_async_warmup())
        print("[LLM] Warmup complete.")


async def _async_warmup() -> None:
    global _tokenizer
    sampling = SamplingParams(max_tokens=8, temperature=0.6, top_p=0.95)
    rid = "warmup-" + uuid.uuid4().hex[:8]
    async for _ in _engine.generate("Hi.", sampling, request_id=rid):
        pass
    # get_tokenizer() is async in vLLM < 0.7, sync in vLLM >= 0.7
    result = _engine.get_tokenizer()
    _tokenizer = await result if asyncio.iscoroutine(result) else result


def is_ready() -> bool:
    return _engine is not None


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def strip_thinking(text: str) -> str:
    return _THINK_PREFIX_RE.sub("", text, count=1).strip()


def extract_thinking(text: str) -> str:
    m = _THINK_PREFIX_RE.match(text)
    return m.group(0) if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING
# ─────────────────────────────────────────────────────────────────────────────
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
    if temperature is not None: base["temperature"] = temperature
    if top_p      is not None: base["top_p"]       = top_p
    if top_k      is not None: base["top_k"]       = top_k
    return SamplingParams(
        **base,
        presence_penalty=PRESENCE_PENALTY if presence_penalty is None else presence_penalty,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHAT MESSAGE BUILDING + IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────
def build_chat_messages(
    system_prompt: Optional[str],
    user_messages: List[Dict[str, Any]],
    image_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
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
    from PIL import Image
    from urllib.request import urlopen
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


# ─────────────────────────────────────────────────────────────────────────────
# CHAT GENERATION
# ─────────────────────────────────────────────────────────────────────────────
async def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        result = _engine.get_tokenizer()
        _tokenizer = await result if asyncio.iscoroutine(result) else result
    return _tokenizer


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
    if _engine is None:
        raise RuntimeError("LLM not initialised — call initialize() first.")

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


# ─────────────────────────────────────────────────────────────────────────────
# HANDLER  (replicates handle_llm from rp_handler.py — no RunPod dependency)
# ─────────────────────────────────────────────────────────────────────────────
async def handle_request(input_data: Dict[str, Any]) -> Dict[str, Any]:
    if not is_ready():
        return {"status": "error", "error": "LLM not initialized."}

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
        return {"status": "error", "error": "Either `messages` or `text_prompt` is required."}

    chat_messages = build_chat_messages(
        system_prompt=system_prompt,
        user_messages=messages,
        image_source=image_source,
    )
    try:
        result = await chat_generate(
            chat_messages,
            thinking_mode=enable_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            return_thinking=return_thinking,
        )
        message: Dict[str, Any] = {"role": "assistant", "content": result["response"]}
        if return_thinking and result["thinking"]:
            message["thinking"] = result["thinking"]
        return {
            "status": "success",
            "response": {
                "choices": [{"message": message, "finish_reason": result["finish_reason"], "index": 0}],
                "model":   MODEL_ID,
                "usage":   {
                    "prompt_tokens":     result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens":      result["prompt_tokens"] + result["completion_tokens"],
                },
                "request_id": result["request_id"],
            },
            "_timing": result,  # kept for stress-test stats
        }
    except Exception as e:
        import traceback as _tb
        print(_tb.format_exc())
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — timed request wrapper + stats printer
# ─────────────────────────────────────────────────────────────────────────────
async def timed_request(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Wraps handle_request and adds wall-clock latency + token counts."""
    t0  = time.perf_counter()
    res = await handle_request(input_data)
    lat = time.perf_counter() - t0
    res["_lat_s"] = lat
    if res["status"] == "success":
        usage = res["response"]["usage"]
        res["_completion_tokens"] = usage["completion_tokens"]
        res["_prompt_tokens"]     = usage["prompt_tokens"]
    else:
        res["_completion_tokens"] = 0
        res["_prompt_tokens"]     = 0
    return res


def _percentile(sorted_data: List[float], p: float) -> float:
    if not sorted_data:
        return float("nan")
    k = (len(sorted_data) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return sorted_data[f] if f == c else sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def print_stats(label: str, results: List[Dict[str, Any]], wall_s: float) -> None:
    total    = len(results)
    ok_res   = [r for r in results if r["status"] == "success"]
    n_ok     = len(ok_res)
    n_fail   = total - n_ok
    lats     = sorted(r["_lat_s"] for r in results)
    comp_tok = sum(r["_completion_tokens"] for r in results)
    prom_tok = sum(r["_prompt_tokens"]     for r in results)

    print(f"\n  ┌─ {label}")
    print(f"  │  requests      : {total}  ({n_ok} ok, {n_fail} failed)")
    print(f"  │  wall time     : {wall_s:.2f}s")
    print(f"  │  latency (s)   : "
          f"min={min(lats):.2f}  "
          f"p50={_percentile(lats,50):.2f}  "
          f"p95={_percentile(lats,95):.2f}  "
          f"p99={_percentile(lats,99):.2f}  "
          f"max={max(lats):.2f}")
    print(f"  │  throughput    : {comp_tok/wall_s:.1f} completion tok/s  |  "
          f"{n_ok/wall_s:.2f} req/s")
    print(f"  │  tokens total  : {prom_tok} prompt  +  {comp_tok} completion")
    if n_fail:
        errors = set(r["error"] for r in results if r["status"] == "error")
        print(f"  │  errors        : {errors}")
    print(f"  └{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; BOLD = "\033[1m"; RESET = "\033[0m"
_passed = _failed = 0

def ok(name):
    global _passed; _passed += 1
    print(f"  {GREEN}PASS{RESET}  {name}")

def fail(name, detail=""):
    global _failed; _failed += 1
    print(f"  {RED}FAIL{RESET}  {name}" + (f"\n         {detail}" if detail else ""))

def section(title: str):
    print(f"\n{BOLD}{'═'*62}{RESET}\n{BOLD}  {title}{RESET}\n{BOLD}{'═'*62}{RESET}")

def chk(cond, name, detail=""):
    ok(name) if cond else fail(name, detail)


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTNESS TESTS
# ─────────────────────────────────────────────────────────────────────────────
async def correctness_tests():

    section("1 · Engine initialisation")
    initialize()
    chk(is_ready(), "engine is ready after initialize()")

    # ── 2. text_prompt + thinking ON + return_thinking ───────────────────────
    section("2 · text_prompt | thinking=ON | return_thinking=True")
    r = await handle_request({
        "text_prompt":    "What is 2 + 2? Give a one-sentence answer.",
        "enable_thinking": True,
        "return_thinking": True,
        "max_tokens":      512,
    })
    chk(r["status"] == "success", "status=success")
    choice = r["response"]["choices"][0]
    resp   = choice["message"]["content"]
    print(f"  response : {resp[:150]}")
    chk(isinstance(resp, str) and len(resp) > 0,           "response is non-empty")
    chk("<think>" not in resp and "</think>" not in resp,   "think tags stripped from response")
    chk("thinking" in choice["message"],                    "thinking field present")
    chk(len(choice["message"].get("thinking", "")) > 0,    "thinking field is non-empty")
    chk(r["response"]["usage"]["completion_tokens"] > 0,   "completion_tokens > 0")
    chk(r["response"]["model"] == MODEL_ID,                 "model ID correct")

    # ── 3. messages list + system_prompt + thinking OFF ──────────────────────
    section("3 · messages list | system_prompt | thinking=OFF")
    r = await handle_request({
        "messages":       [{"role": "user", "content": "Say hello in exactly 3 words."}],
        "system_prompt":  "You are a concise assistant.",
        "enable_thinking": False,
        "max_tokens":      64,
    })
    chk(r["status"] == "success", "status=success")
    resp = r["response"]["choices"][0]["message"]["content"]
    print(f"  response : {resp[:150]}")
    chk(len(resp) > 0,              "response non-empty")
    chk("<think>" not in resp,      "no think tags in non-thinking mode")
    chk("thinking" not in r["response"]["choices"][0]["message"], "thinking field absent")

    # ── 4. Multi-turn context recall ─────────────────────────────────────────
    section("4 · Multi-turn context recall")
    r = await handle_request({
        "messages": [
            {"role": "user",      "content": "My name is Alex."},
            {"role": "assistant", "content": "Hello Alex, nice to meet you!"},
            {"role": "user",      "content": "What is my name?"},
        ],
        "enable_thinking": False,
        "max_tokens":      64,
    })
    chk(r["status"] == "success", "status=success")
    resp = r["response"]["choices"][0]["message"]["content"]
    print(f"  response : {resp[:150]}")
    chk("Alex" in resp or "alex" in resp.lower(), "model recalls name from prior turn")

    # ── 5. Error: no input ───────────────────────────────────────────────────
    section("5 · Error handling — missing input")
    r = await handle_request({})
    chk(r["status"] == "error", "empty input → error status")
    chk(len(r.get("error", "")) > 0, "error message is non-empty")

    # ── 6. Parameter overrides ───────────────────────────────────────────────
    section("6 · Parameter overrides (temperature, top_k, max_tokens)")
    r = await handle_request({
        "text_prompt":    "Name a colour.",
        "enable_thinking": False,
        "temperature":     0.1,
        "top_k":           10,
        "max_tokens":      16,
    })
    chk(r["status"] == "success", "status=success")
    resp = r["response"]["choices"][0]["message"]["content"]
    print(f"  response : {resp[:150]}")
    chk(r["response"]["usage"]["completion_tokens"] <= 16, "max_tokens=16 respected")

    # ── 7. Raw output — think tags present in raw, stripped in response ───────
    section("7 · Raw output: </think> in raw, absent in response")
    chat = build_chat_messages(None, [{"role": "user", "content": "Is 17 prime? Answer in one word."}])
    raw_result = await chat_generate(chat, thinking_mode=True, max_tokens=512, return_thinking=True)
    print(f"  raw (200 chars) : {raw_result['raw'][:200]!r}")
    print(f"  response        : {raw_result['response']!r}")
    chk("</think>" in raw_result["raw"],          "</think> in raw output (thinking active)")
    chk("</think>" not in raw_result["response"], "</think> absent in cleaned response")
    chk(raw_result["finish_reason"] in ("stop", "length"), "finish_reason valid")


# ─────────────────────────────────────────────────────────────────────────────
# STRESS TESTS
# ─────────────────────────────────────────────────────────────────────────────

# Diverse prompts — spread across subject areas so prefix cache doesn't skew results
_PROMPTS_SHORT = [
    "What is the boiling point of water in Celsius?",
    "Name the longest river in Africa.",
    "Who wrote Hamlet?",
    "What is the speed of light in m/s?",
    "Name the largest planet in the solar system.",
    "What language is spoken in Brazil?",
    "How many sides does a hexagon have?",
    "What is the chemical symbol for gold?",
    "Name the capital of Japan.",
    "What year did World War II end?",
    "What is the square root of 144?",
    "Name the author of 1984.",
    "What gas do plants absorb during photosynthesis?",
    "How many bones are in the adult human body?",
    "What currency does Japan use?",
    "What is the smallest prime number?",
    "Name the largest ocean on Earth.",
    "What is Newton's first law of motion?",
    "Who painted the Mona Lisa?",
    "What is H2O commonly known as?",
    "What continent is Egypt in?",
    "Name the three primary colours.",
    "What organ pumps blood through the body?",
    "What is the SI unit of force?",
    "How many degrees are in a right angle?",
    "What is the hardest natural substance?",
    "Name the tallest mountain on Earth.",
    "What does DNA stand for?",
    "What planet is closest to the Sun?",
    "How many seconds are in an hour?",
    "What is the freezing point of water in Fahrenheit?",
    "Name the powerhouse of the cell.",
]

_PROMPTS_MEDIUM = [
    "Explain the concept of supply and demand in 2-3 sentences.",
    "What are the main differences between RAM and ROM?",
    "Briefly describe how a neural network learns.",
    "What is the difference between HTTP and HTTPS?",
    "Explain what recursion means in programming.",
    "What causes the seasons on Earth?",
    "Describe how vaccines work in 2-3 sentences.",
    "What is the difference between a compiler and an interpreter?",
]

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer factually and concisely."
)


async def stress_a_max_concurrency():
    """
    Wave A — Fire MAX_NUM_SEQS requests simultaneously.
    All requests arrive at t=0 and compete for the same batch slots.
    This is the tightest test of the continuous-batching scheduler.
    """
    section(f"STRESS A · Max-concurrency burst ({MAX_NUM_SEQS} simultaneous requests)")
    prompts = (_PROMPTS_SHORT * 4)[:MAX_NUM_SEQS]
    tasks   = [
        timed_request({
            "text_prompt":    p,
            "system_prompt":  _SYSTEM_PROMPT,
            "enable_thinking": False,
            "max_tokens":      128,
        })
        for p in prompts
    ]
    t0      = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall    = time.perf_counter() - t0

    print_stats(f"Max-concurrency burst  n={MAX_NUM_SEQS}", results, wall)
    n_ok = sum(1 for r in results if r["status"] == "success")
    chk(n_ok == MAX_NUM_SEQS, f"all {MAX_NUM_SEQS} concurrent requests succeeded")
    chk(
        all("<think>" not in r["response"]["choices"][0]["message"]["content"]
            for r in results if r["status"] == "success"),
        "no think tags leaked into any response",
    )
    return results


async def stress_b_sustained_load():
    """
    Wave B — 3 back-to-back waves of MAX_NUM_SEQS each (96 total requests).
    Models sustained traffic: the KV cache, prefix cache, and scheduler
    all stay active across waves.
    """
    n_waves = 3
    total   = n_waves * MAX_NUM_SEQS
    section(f"STRESS B · Sustained load  ({n_waves} waves × {MAX_NUM_SEQS} = {total} requests)")
    all_results: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for wave in range(n_waves):
        prompts = (_PROMPTS_SHORT * 4)[wave * MAX_NUM_SEQS % len(_PROMPTS_SHORT * 4):
                                       (wave + 1) * MAX_NUM_SEQS % len(_PROMPTS_SHORT * 4) + MAX_NUM_SEQS]
        prompts = (prompts * 2)[:MAX_NUM_SEQS]
        tasks   = [
            timed_request({
                "text_prompt":    p,
                "system_prompt":  _SYSTEM_PROMPT,
                "enable_thinking": False,
                "max_tokens":      128,
            })
            for p in prompts
        ]
        t_wave    = time.perf_counter()
        wave_res  = await asyncio.gather(*tasks)
        wave_wall = time.perf_counter() - t_wave
        all_results.extend(wave_res)
        n_ok = sum(1 for r in wave_res if r["status"] == "success")
        print(f"  wave {wave+1}/{n_waves}: {n_ok}/{MAX_NUM_SEQS} ok  "
              f"wall={wave_wall:.2f}s  "
              f"tok/s={sum(r['_completion_tokens'] for r in wave_res)/wave_wall:.1f}")

    wall = time.perf_counter() - t_start
    print_stats(f"Sustained load  n={total}", all_results, wall)
    n_ok = sum(1 for r in all_results if r["status"] == "success")
    chk(n_ok == total, f"all {total} requests in sustained load succeeded")
    return all_results


async def stress_c_mixed_modes():
    """
    Wave C — Equal split of thinking=True and thinking=False requests,
    all concurrent. Tests the scheduler with heterogeneous sampling params.
    """
    n = MAX_NUM_SEQS
    section(f"STRESS C · Mixed thinking/non-thinking  ({n} concurrent requests, 50/50 split)")
    tasks = []
    for i, p in enumerate((_PROMPTS_SHORT * 2)[:n]):
        thinking = (i % 2 == 0)
        tasks.append(timed_request({
            "text_prompt":    p,
            "enable_thinking": thinking,
            "max_tokens":      256,
        }))

    t0      = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall    = time.perf_counter() - t0

    print_stats(f"Mixed modes  n={n}", results, wall)
    n_ok = sum(1 for r in results if r["status"] == "success")
    chk(n_ok == n, f"all {n} mixed-mode requests succeeded")
    return results


async def stress_d_prefix_cache():
    """
    Wave D — All requests share an identical system prompt.
    With prefix caching ON, requests after the first should see a faster
    time-to-first-token because the prompt KV blocks are reused.
    We fire them in two rounds and compare mean latency.
    """
    section("STRESS D · Prefix-cache effectiveness (shared system prompt, 2 rounds)")
    shared_sys = (
        "You are an expert tutor specialising in STEM subjects. "
        "You give concise, accurate answers aimed at university-level students. "
        "Always be direct and avoid filler phrases."
    )
    prompts = (_PROMPTS_SHORT * 2)[:16]

    async def _run_round(label: str) -> List[Dict[str, Any]]:
        tasks = [
            timed_request({
                "text_prompt":    p,
                "system_prompt":  shared_sys,
                "enable_thinking": False,
                "max_tokens":      64,
            })
            for p in prompts
        ]
        t0  = time.perf_counter()
        res = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
        print_stats(label, res, wall)
        return res

    round1 = await _run_round("Round 1 (cold prefix cache)")
    round2 = await _run_round("Round 2 (warm prefix cache — should be faster)")

    mean1 = statistics.mean(r["_lat_s"] for r in round1)
    mean2 = statistics.mean(r["_lat_s"] for r in round2)
    print(f"\n  Round 1 mean latency: {mean1:.2f}s")
    print(f"  Round 2 mean latency: {mean2:.2f}s")
    speedup = (mean1 - mean2) / mean1 * 100
    print(f"  Speedup from prefix cache: {speedup:.1f}%")
    chk(
        sum(1 for r in round2 if r["status"] == "success") == len(prompts),
        "all round-2 requests succeeded",
    )


async def stress_e_longer_outputs():
    """
    Wave E — Fewer requests but with more output tokens (512).
    Stresses the KV cache and decode throughput under long generations,
    mirroring the production default of max_tokens=2048.
    """
    n = min(8, MAX_NUM_SEQS)
    section(f"STRESS E · Long outputs  ({n} concurrent, max_tokens=512, thinking=ON)")
    medium = (_PROMPTS_MEDIUM * 4)[:n]
    tasks = [
        timed_request({
            "text_prompt":    p,
            "enable_thinking": True,
            "return_thinking": False,
            "max_tokens":      512,
        })
        for p in medium
    ]
    t0      = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall    = time.perf_counter() - t0

    print_stats(f"Long outputs  n={n}", results, wall)
    n_ok = sum(1 for r in results if r["status"] == "success")
    chk(n_ok == n, f"all {n} long-output requests succeeded")
    chk(
        all(r["response"]["usage"]["completion_tokens"] > 50
            for r in results if r["status"] == "success"),
        "each response has >50 completion tokens",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
async def run_all_tests():
    # ── Correctness suite ────────────────────────────────────────────────────
    await correctness_tests()

    # ── Stress suite ─────────────────────────────────────────────────────────
    section("STRESS TESTS")
    print(f"  Model: {MODEL_ID}")
    print(f"  MAX_NUM_SEQS: {MAX_NUM_SEQS}  |  SPECULATIVE: {SPECULATIVE_ENABLED}  "
          f"|  PREFIX_CACHE: {ENABLE_PREFIX_CACHE}")

    await stress_a_max_concurrency()
    await stress_b_sustained_load()
    await stress_c_mixed_modes()
    await stress_d_prefix_cache()
    await stress_e_longer_outputs()

    # ── Final summary ────────────────────────────────────────────────────────
    total = _passed + _failed
    colour = GREEN if _failed == 0 else RED
    print(f"\n{colour}{BOLD}{'═'*62}{RESET}")
    print(f"{colour}{BOLD}  Results: {_passed}/{total} passed,  {_failed} failed{RESET}")
    print(f"{colour}{BOLD}{'═'*62}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(run_all_tests())
else:
    print("In Colab, run:  await run_all_tests()")
