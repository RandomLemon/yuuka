# Serving and Scheduling

## 1. Scope

This document covers the request path above the model executor: admission,
batching, paged KV cache, HIP Graph replay, APIs, and observability. It assumes
the single-request GPU path is already correct and measurable.

Current implementation:

- One loaded model served by a continuous scheduler with configurable active
  request, queue, batch, chunk, timeout, and concurrency limits.
- Paged KV cache: a reference-counted block allocator, per-request immutable
  block tables, shared prefix blocks, and explicit release on cancellation and
  failure. The current pool is FP32; FP16 and quantized KV remain extension
  points.
- Prefix cache keyed by the model identity, block size, and exact token prefix.
  Entries retain KV blocks, the recurrent state snapshot, and the prompt logits.
  Reuse is longest-prefix and collision-safe (full token comparison), with LRU
  eviction under block pressure.
- Continuous batching at the scheduler level: prefill and decode steps are
  interleaved between requests, and each iteration records its batch
  composition. One model stream executes one request plan at a time; mixed
  prefill/decode shapes are not submitted to a single kernel batch.
- HIP Graph capture and replay for decode, using device-resident token and
  position parameters. Graph capture is opt-in because the Phase 4
  measurements on Qwen3.5-4B Q4_K_M showed no material decode gain at 256
  context; the plain stream path is the default and capture/replay falls back
  to it on any failure.
- OpenAI-compatible `POST /v1/chat/completions` with SSE streaming, structured
  errors, stop strings, and a strict supported-parameter subset.
- `/healthz`, `/metrics`, `/v1/models`, and `/debug/pprof/` endpoints.
- Request cancellation from the HTTP context, request and queue timeouts,
  queue-depth backpressure, and graceful shutdown.

The current throughput mode is decode-priority interleaving. The scheduler does
not yet compose one mixed-shape kernel batch across requests, so aggregate
decode throughput stays close to the single-request rate; concurrency improves
admission and TTFT fairness rather than token throughput.

## 2. Serving architecture

```text
Request queue
      |
Admission and KV budget
      |
Continuous scheduler
   /            \
Chunked prefill  Decode
   \            /
GPU execution (prefill or decode plan)
      |
Sampling + streaming
      |
Response / cancellation / cleanup
```

The scheduler owns admission and batch composition. The executor owns the
model-specific prefill/decode loop. The memory manager owns KV blocks and
scratch. No layer may bypass admission to allocate KV blocks directly.

Implementation mapping:

| Layer | Package |
| --- | --- |
| HTTP API, SSE, metrics | `cmd/yuuka-server` |
| Request state machine, sampling, stops, prefix insertion | `serving` |
| Admission, continuous batching, timeouts, lifecycle metrics | `scheduler` |
| Block allocator, block table, reference counting, prefix cache | `memory` |
| CPU model execution | `model/qwen35`, `serving.Qwen35Backend` |
| HIP model execution, paged kernels, graph replay | `engine/gpu`, `serving.Qwen35GPUBackend` |

## 3. Request lifecycle

| State | Meaning | Exit conditions |
| --- | --- | --- |
| **Queued** | Accepted but not yet admitted | Admitted, rejected, timed out, client cancelled |
| **Prefill** | Prompt chunks are being processed | Prefill complete, cancelled, failed |
| **Decode** | Tokens are being generated | Stop condition, max tokens, cancelled, failed |
| **Finishing** | Stream flush and prefix insertion | Response completed |
| **Closed** | Resources released | Terminal state |

Each request carries a stable ID (`chatcmpl-<n>` in the server) used by logs,
metrics, cancellation, and cleanup. A request that fails or is cancelled
releases its block table, task, and stream state exactly once. Cancellation is
observed between model steps; a prefill chunk or decode step already submitted
to the device completes first.

## 4. Admission and budgets

Admission prevents overcommit and keeps latency predictable. It accounts for:

- **Token budget:** prompt plus requested output tokens, bounded by the
  configured context length.
- **KV budget:** blocks required for prompt plus output positions, checked
  against the allocator total before queueing. A prefix hit can reduce the
  actual allocation but does not reduce the conservative admission estimate.
- **Concurrency limit:** active requests (`--max-concurrency`) and queue depth
  (`--max-queue`).
- **Scratch and graph workspace:** per-session prefill plans, trace buffers, and
  graph executables.

Implementation policy:

- A request whose prompt plus output exceeds the context is rejected with
  `context_length_exceeded`.
- A request that cannot fit the KV pool even with an empty queue is rejected
  with `insufficient_capacity`.
- A full queue is rejected with `server_overloaded` and `Retry-After: 1`.
- Output tokens are clamped to the remaining context, matching the CLI.
- A request that fails allocation when its task is built fails alone; the
  server stays alive and the scheduler slot is released.

### 4.1 Server limits

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ctx` | 4096 | Maximum prompt plus output positions per request |
| `--block-size` | 16 | KV block size in tokens |
| `--kv-blocks` | derived | Total KV blocks. The default is `(ceil(ctx/block-size) + 1) * max-concurrency`, including one block of headroom per request for prefix copies and cache entries |
| `--max-concurrency` | 4 | Maximum active requests |
| `--max-queue` | 64 | Maximum pending requests |
| `--prefill-chunk` | 512 | Prompt tokens per prefill opportunity |
| `--request-timeout` | 5m | Per-request deadline, including queue wait |
| `--queue-timeout` | 30s | Maximum time a request may remain queued |
| `--prefix-cache` | true | Enable prompt prefix caching |
| `--prefix-entries` | 64 | Maximum cached prompt prefixes |
| `--prefix-blocks` | 0 (pool bound) | Maximum blocks held by the prefix cache |
| `--graph` | false | Enable HIP Graph capture/replay with fallback |
| `--model-key` | file SHA-256 | Override the prefix-cache model identity |

The HTTP request body limit is 8 MiB. Limits are reported by `/healthz` and
`/metrics`, and rejected requests include an opaque request ID.

## 5. Scheduling modes

| Mode | Goal | Batch policy |
| --- | --- | --- |
| **Latency** | Minimize TTFT and TPOT for one user | Small batches; prioritize the active request; avoid unnecessary prefill interleaving |
| **Throughput** | Maximize aggregate tokens/s | Continuous batching; reuse weights across requests; interleave prefill and decode within budget |

The mode is a server-level or request-class configuration. It must not change
model semantics or numerical results.

Current status: one decode-priority continuous mode. Each scheduler iteration
first runs up to `DecodePerCycle` ready decode steps in round-robin order, then
one prefill opportunity (configurable) so a long prompt cannot monopolize the
stream. Newly admitted requests join the rotation immediately. A separate
latency/throughput mode switch is not exposed yet.

### 5.1 Scheduling policy

- Decode steps have priority over new prefill work to protect TPOT.
- Prefill is split into chunks; a long prompt cannot occupy the GPU for an
  unbounded period while other requests are decoding.
- A newly admitted request receives a bounded prefill opportunity, then yields
  to pending decode work.
- Ready decode requests are rotated so every request gets a step.
- Metrics and per-request results record queue wait, prefill, decode, and
  sampling separately.

## 6. Continuous batching and chunked prefill

Continuous batching admits new requests between decode steps rather than
waiting for the entire batch to finish. The batch is recomposed as requests
arrive, complete, or cancel. Every iteration records its decode and prefill
request IDs through the scheduler's `OnStep` hook.

Requirements and current behavior:

- Batch composition is explicit and observable.
- Per-request sequence lengths and block tables are passed to the kernels.
- There is no padded lane because requests are not fused into one kernel batch;
  each request's plan sees only its own table and positions.
- A request removal does not invalidate the KV blocks of other requests.
- Chunked prefill uses the same causal attention semantics as full prefill; the
  paged prefill kernel enforces the causal limit across chunk boundaries.
- Prefill and decode alternate rather than sharing a mixed-shape step. The
  kernels do not yet support a mixed prefill/decode batch, which is why
  aggregate throughput does not scale with concurrency.
- Prefill and decode are both submitted to the same explicit model stream, so
  ordering is deterministic.

## 7. Paged KV cache

### 7.1 Configuration

- FP32 keys and values. FP16 KV and quantized KV remain extension points because
  the accepted numerical baselines use FP32.
- Configurable block size; the server and benchmark default to 16 tokens.
- One allocator per model with a fixed block count derived from context and
  concurrency, or set explicitly with `--kv-blocks`.
- Per-request block table mapping logical positions to physical blocks.
- The model and kernels address KV only through the block table; model code
  never sees physical block placement.

### 7.2 Components

| Component | Responsibility | Implementation |
| --- | --- | --- |
| **Block allocator** | Allocate, free, and reuse fixed-size blocks; track free capacity | `memory.Allocator` |
| **Request block table** | Map logical positions to physical blocks for each layer | `memory.Table` |
| **Reference counting** | Share blocks for prefix caching; free only at the last release | `memory.Allocator` reference counts |
| **Prefix cache** | Identify and reuse common prompt prefixes | `memory.PrefixCache` |
| **Cancellation cleanup** | Release all blocks owned by a cancelled or failed request | `scheduler` closes the task, which releases the table |
| **Eviction** | Reclaim cached blocks under memory pressure | LRU callback installed on the allocator |

### 7.3 Invariants

- A block is freed exactly once; double release is an error.
- A block is not returned to the free list while any request or cache entry
  still references it.
- Kernel reads never cross a block boundary without the block table; the paged
  attention and KV-write kernels resolve every position through the table.
- The block table is fixed for the duration of a request. All required blocks
  are allocated at admission, so a submitted step never reallocates the table.
- KV writes for a token occur before any later decode step reads that token;
  both run on the same explicit stream.
- A request cannot observe blocks belonging to another request. Blocks shared
  with the prefix cache are read-only; a partially shared last block is copied
  before the request writes past the cached prefix.

### 7.4 Prefix cache

A prefix-cache entry represents one complete prompt processed by an earlier
request.

- **Key:** SHA-256 over a cache version tag, the model identity supplied by the
  server (by default the SHA-256 of the GGUF file), the block size, the token
  count, and every token. Lookup additionally compares the full token slice, so
  a hash collision cannot produce incorrect reuse.
- **Payload:** block references covering the prompt, the recurrent
  convolution/matrix state snapshot at the prompt boundary, and the logits for
  the last prompt token. The logits let a full-prompt hit sample immediately.
- **Partial final block:** when the prompt is not block-aligned, the cache owns
  the original partial block and the request receives a private copy, so later
  decode writes cannot corrupt the cached KV.
- **Eviction:** least-recently-used. The allocator invokes the cache when no
  free block remains; cache entries release their own references while
  request-held references keep shared blocks alive. Entry and block caps are
  configurable with `--prefix-entries` and `--prefix-blocks`.
- **Insertion:** after a completed prompt, the serving manager splits the prompt
  prefix out of the request table and inserts it. Cache failure or allocation
  pressure never changes the request's generated output.

Hit rate and cache occupancy are reported by `/metrics`.

## 8. HIP Graph

The plain HIP stream path is the default and remains correct without graph
capture. Replay is opt-in (`--graph`) and was introduced only after the decode
host path was measured.

Behavior:

- The decode plan is captured after its first eligible decode step and
  instantiated once per session.
- The captured graph addresses weights, activations, KV pools, the session
  block table, and the dynamic parameter buffer, all of which have stable
  addresses for the session lifetime.
- Kernel arguments do not change between replays. A two-element device buffer
  supplies the current token and position; `yuuka_dynamic_set` uploads it on
  the same stream before `hipGraphLaunch`.
- Before replay the runtime verifies that the session is open, the position is
  inside capacity, the block table covers the next position, and the dynamic
  buffer is valid.
- If capture, parameter upload, or replay fails, the graph is destroyed and the
  same step is submitted through the plain plan. A capture failure disables
  graph use for that session; it never corrupts output.
- Prefill plans are not captured; the decode shape is the stable replay class.
- Graph resources are released with the session before the model.

Measured Phase 4 result: decode time with and without replay is effectively
identical for Qwen3.5-4B Q4_K_M at 256 context (1.533 s vs 1.530 s for 16
tokens), so the server default is off. The implementation remains for stable
shapes where host launch overhead is measured to matter.

## 9. API surface

### 9.1 CLI

The CLI supports loading a GGUF model, interactive generation, inspection, and
benchmark modes. Serving uses `yuuka-server`.

### 9.2 HTTP

The server targets an OpenAI-compatible subset and rejects unknown JSON fields:

| Surface | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Chat generation, streaming or non-streaming |
| `GET /v1/models` | Loaded model metadata and capabilities |
| `GET /healthz` | Liveness, queue/active counts, KV occupancy |
| `GET /metrics` | Prometheus text metrics |
| `GET /debug/pprof/*` | Go runtime profiling endpoints |

Supported chat parameters: `model`, string or text-part `messages`,
`max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`,
`stream`, `stop` (string or array), `stop_token_ids`, `enable_thinking`,
`chat_template_kwargs.enable_thinking`, and `user` (ignored). Unknown JSON
fields are rejected. `n != 1`, nonzero penalties, logprobs, `response_format`,
`tools`, and `stream_options` are rejected explicitly rather than ignored.

Streaming uses server-sent events with an initial assistant role chunk,
incremental content deltas, a terminal chunk with `finish_reason`, and
`data: [DONE]`. A cancelled client context stops generation, releases the
request, and is reported in cancellation metrics.

### 9.3 Error model

Errors are JSON objects containing `message`, `type`, `code`, `request_id`, and
`retryable`. Streaming errors use the same payload inside a `data:` event
followed by `[DONE]`.

| Code | Status | Meaning |
| --- | --- | --- |
| `invalid_request` | 400 | Malformed JSON, empty messages, invalid content |
| `unsupported_parameter` | 400 | A known OpenAI parameter outside the supported subset |
| `context_length_exceeded` | 400 | Prompt plus output exceeds the configured context |
| `model_not_found` | 404 | Requested model identifier is not loaded |
| `server_overloaded` | 429 | Queue depth limit reached; retryable |
| `insufficient_capacity` | 503 | KV pool cannot admit the request; retryable |
| `request_timeout` | 504 | Request deadline expired |
| `client_closed` | 499 | Client cancelled the request |
| `generation_error` | 500 | Model, device, or streaming failure |

## 10. Observability

The scheduler records:

- queue depth, active requests, last and maximum batch size, maximum active;
- queue wait, prefill, decode, and sampling totals;
- TTFT and TPOT P50/P95/P99 and maximum over a bounded sample window;
- prompt and generated token totals;
- submissions, admissions, completions, failures, cancellations, timeouts,
  queue-full rejections, and capacity rejections.

The KV allocator reports total, used, free, shared, and block size. The prefix
cache reports entries, blocks, hits, misses, insertions, and evictions.

`/metrics` exposes these as `yuuka_*` Prometheus counters and gauges, including
`yuuka_ttft_seconds{quantile="0.5|0.95|0.99"}`, `yuuka_tpot_seconds`,
`yuuka_queue_wait_seconds_total`, `yuuka_prefill_seconds_total`,
`yuuka_decode_seconds_total`, and `yuuka_sampling_seconds_total`.

Logging includes the request ID, finish reason, token counts, phase timings,
and prefix hit; prompt contents are not logged.

## 11. Backpressure and failure handling

- Queue-depth and concurrency limits are enforced before task construction.
- Capacity is checked before KV allocation; allocation failure fails one
  request and releases its blocks and task.
- The model stream is explicit; a device error fails the active request and is
  surfaced as `generation_error`. The server keeps running while the device
  remains usable.
- On shutdown, HTTP stops accepting connections, the scheduler stops admission,
  and active requests drain for up to 10 seconds before cancellation.
- A cancelled or failed request always releases its task, block table, and
  graph resources. The server never returns success for a failed generation.

## 12. Deferred: multiple models

Model unload and multi-model management are deferred until the serving path is
accepted. The current server loads exactly one model for its lifetime and uses
the model identity in every prefix-cache key so a future multi-model server
cannot mix cache entries.
