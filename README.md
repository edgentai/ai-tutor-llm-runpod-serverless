# ai-tutor-llm-runpod-serverless

Single-tenant Qwen3.6-27B LLM service for RunPod serverless. Replaces the
`general_purpose_llm_job` route from the previous combined handler.

## Layout

```
handler.py        # RunPod entrypoint
models/llm.py     # vLLM wrapper (YaRN, prefix caching, multimodal)
Dockerfile        # vLLM nightly + minimal deps
requirements.txt  # runpod, hf, transformers, sentencepiece, protobuf
test_input.json   # sample payloads
```

## Input contract

`event["input"]` accepts:

| field            | type       | default | notes                                          |
|------------------|------------|---------|------------------------------------------------|
| messages         | list       | —       | `[{"role":..., "content":...}]` (preferred)    |
| text_prompt      | str        | —       | flat shorthand if `messages` not provided      |
| system_prompt    | str        | ""      |                                                |
| image_source     | str        | None    | http(s) URL or `data:image/...;base64,...`     |
| max_tokens       | int        | 2048    |                                                |
| temperature      | float      | preset  | overrides per-mode preset                      |
| top_p / top_k    | float/int  | preset  | overrides per-mode preset                      |
| enable_thinking  | bool       | True    | toggles Qwen3 thinking template                |
| presence_penalty | float      | 1.0     |                                                |
| return_thinking  | bool       | False   | returns the reasoning trace too                |

Output is OpenAI-compatible: `{status, response: {choices, model, usage}}`.

## Env vars

| var                       | default                | notes                                              |
|---------------------------|------------------------|----------------------------------------------------|
| LLM_MODEL_ID              | Qwen/Qwen3.6-27B       | HuggingFace repo id                                |
| LLM_GPU_MEM               | 0.92                   | single-tenant A100-80GB; lower if sharing GPU      |
| LLM_MAX_MODEL_LEN         | 131072                 | YaRN-extended ctx                                  |
| LLM_NATIVE_CTX            | 32768                  | model's native ctx                                 |
| LLM_YARN_FACTOR           | 4.0                    | only applied if MAX > NATIVE                       |
| LLM_MAX_SEQS              | 32                     | concurrent batch ceiling                           |
| LLM_PREFIX_CACHE          | 1                      | 0/1                                                |
| LLM_CHUNKED_PREFILL       | 1                      | 0/1                                                |
| LLM_IMAGES_PER_PROMPT     | 4                      |                                                    |
| LLM_ENFORCE_EAGER         | 0                      | 0=CUDA graphs ON (faster decode, slower cold start)|
| LLM_WARMUP                | 1                      | 0/1                                                |
| LLM_SPECULATIVE           | 1                      | 1=n-gram speculative decoding ON                   |
| LLM_SPEC_NUM_TOKENS       | 5                      | draft tokens per step                              |
| LLM_SPEC_LOOKUP_MAX       | 4                      | longest n-gram match window                        |
| LLM_SPEC_LOOKUP_MIN       | 2                      | shortest n-gram match window                       |
| HF_TOKEN                  | —                      | required for gated models                          |

## Concurrency model

The handler is `async` and uses `vllm.AsyncLLMEngine`, so vLLM's continuous
batcher actually sees concurrent requests and merges them into single
forward passes. The `concurrency_modifier` callback in `handler.py` tells
RunPod each worker is willing to take up to `LLM_MAX_SEQS` jobs in flight
at once — match-by-construction with vLLM's scheduling cap.

What this means in practice on A100-80GB:

| Scenario                          | Latency p50 | Latency p99 | Workers needed |
|-----------------------------------|-------------|-------------|----------------|
| 1 user                            | ~12 s       | ~12 s       | 1              |
| 32 concurrent users (one batch)   | ~18 s       | ~22 s       | 1              |
| 100 concurrent users              | ~25 s       | ~50 s       | 1              |
| 100 concurrent, low-tail target   | ~10 s       | ~15 s       | 4              |

(Numbers assume 1K-prompt / 500-token-output chats. Decode is bandwidth-
bound on A100 at ~28 tok/s/stream; n-gram speculation lifts that to
~40–50 tok/s on RAG-shaped prompts.)

To pre-warm capacity for sustained traffic, set the RunPod endpoint's
**Min Workers** ≥ ceil(expected_concurrent / LLM_MAX_SEQS). Cold starts
otherwise add ~60–90 s (model load + CUDA graph capture + warmup) to
the first request that lands on a fresh worker.

## Sizing

Qwen3.6-27B at bf16 ≈ 54 GB weights. With `LLM_GPU_MEM=0.92` on an A100-80GB
this leaves ~19 GB for KV cache + activations + CUDA graphs — comfortable
headroom for `MAX_NUM_SEQS=32` at long contexts. If your typical context is
short (≤4K tokens) you can push `LLM_MAX_SEQS=64` to double per-worker
concurrency; verify no OOM under your traffic shape first.

## Cold start vs steady state

CUDA graph capture (enforce_eager=0) adds ~30–60s to cold start. With
sustained traffic this is the right default — decode is 10–20% faster.
Sporadic traffic where every request is a fresh worker: set
`LLM_ENFORCE_EAGER=1`.

## Speculative decoding

n-gram speculation is on by default. It costs zero GPU memory — drafts come
from prompt-tail n-gram matches against the running generation, verified in
one forward pass. Wins big (1.5–2×) on RAG / structured output / repeated
patterns; mild win (1.1–1.3×) on free-form chat. Disable only if you suspect
a vLLM-version-specific bug: `LLM_SPECULATIVE=0`.
