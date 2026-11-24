# Serving and Scheduling

## 1. Scope

This document covers the request path above the model executor: admission,
batching, paged KV cache, HIP Graph replay, APIs, and observability. It assumes
the single-request GPU path is already correct and measurable.

The serving design provides:

- one loaded model served by a continuous scheduler with configurable active
  request, queue, batch, chunk, timeout, and concurrency limits;
- a reference-counted paged KV cache with per-request immutable block tables,
  shared prefix blocks, and explicit release on cancellation and failure;
- a prefix cache keyed by model identity, block size, KV format, and exact token
  prefix, with longest-prefix reuse and LRU eviction;
- iteration-level continuous scheduling, where prefill and decode steps are
  interleaved between requests;
- opt-in HIP Graph capture and replay for decode, with fallback to the plain
  stream path;
- an OpenAI-compatible chat API with SSE streaming, health, metrics, and
  profiling endpoints;
- request cancellation, request and queue timeouts, queue-depth backpressure,
  and graceful shutdown.

The scheduler does not compose one mixed-shape kernel batch across requests.
Aggregate decode throughput therefore stays close to the single-request rate;
concurrency improves admission and TTFT fairness rather than token throughput.

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

A queue-full rejection must be returnable as an HTTP error before the first SSE
event, so streaming starts only after the request has been queued.

## 3. Request lifecycle

| State | Meaning | Exit conditions |
| --- | --- | --- |
| **Queued** | Accepted but not yet admitted | Admitted, rejected, timed out, client cancelled |
| **Prefill** | Prompt chunks are being processed | Prefill complete, cancelled, failed |
| **Decode** | Tokens are being generated | Stop condition, max tokens, cancelled, failed |
| **Finishing** | Stream flush and prefix insertion | Response completed |
| **Closed** | Resources released | Terminal state |

Each request carries a stable ID used by logs, metrics, cancellation, and
cleanup. A request that fails or is cancelled releases its block table, task, and
stream state exactly once. Cancellation is observed between model steps; a
prefill chunk or decode step already submitted to the device completes first.

## 4. Admission and budgets

Admission prevents overcommit and keeps latency predictable. It accounts for:

- **Token budget:** prompt plus requested output tokens, bounded by the
  configured context length.
- **KV budget:** blocks required for prompt plus output positions, checked
  against the configured context before queueing. The context is clamped to the
  allocator capacity when the manager is constructed, so requests that fit the
  context fit the pool in isolation. A prefix hit can reduce the actual
  allocation but does not reduce the conservative estimate.
- **Concurrency limit:** active requests and queue depth.

Scratch and graph workspace are allocated per session after admission; their
sizes are bounded by the model geometry and configured chunk size, not by a
separate admission budget.

Admission policy:

- A request whose prompt alone consumes the whole context is rejected with
  `context_length_exceeded`.
- A request whose block table cannot be allocated, including a pool exhausted by
  concurrent requests or retained prefix entries, is rejected with
  `insufficient_capacity`.
- A full queue is rejected with `server_overloaded` and `Retry-After: 1`.
  Streaming requests receive the HTTP error when rejection happens before the
  first SSE event; errors after streaming starts use an SSE error event.
- Output tokens are clamped to the remaining context; a request is not rejected
  solely because the requested output budget exceeds the remaining positions.
- A request that fails allocation fails alone; the schedule keeps running and the
  slot is released.

### 4.1 Server limits

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ctx` | 4096 | Maximum prompt plus output positions per request |
| `--block-size` | 16 | KV block size in tokens |
| `--kv-blocks` | derived | Total KV blocks; the default includes headroom per request for prefix copies and cache entries |
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

The HTTP request body limit is 8 MiB; larger bodies return HTTP 413 with
`request_too_large`. The decoder accepts exactly one JSON value, so trailing
data after the request object is rejected with `invalid_request`. Effective
limits are reported in the `limits` object of `/healthz` and as `yuuka_*` gauges
on `/metrics`. Rejected requests include an opaque request ID.

## 5. Scheduling modes

| Mode | Goal | Batch policy |
| --- | --- | --- |
| **Latency** | Minimize TTFT and TPOT for one user | Small batches; prioritize the active request; avoid unnecessary prefill interleaving |
| **Throughput** | Maximize aggregate tokens/s | Iteration-level interleaving; interleave prefill and decode within budget. Fused multi-request batching is deferred. |

A mode must not change model semantics or numerical results. The implemented
mode is decode-priority continuous scheduling; a separate latency/throughput
switch is not exposed.

### 5.1 Scheduling policy

- Decode steps have priority over new prefill work to protect TPOT.
- Prefill is split into chunks; a long prompt cannot occupy the GPU for an
  unbounded period while other requests are decoding.
- A newly admitted request receives a bounded prefill opportunity, then yields
  to pending decode work.
- Ready decode requests are rotated so every request gets a step.
- Metrics and per-request results record queue wait, prefill, decode, and
  sampling separately.

## 6. Iteration-level continuous scheduling and chunked prefill

Iteration-level continuous scheduling admits new requests between decode steps
rather than waiting for the entire batch to finish. The active set is recomposed
as requests arrive, complete, or cancel.

Requirements:

- Batch composition is explicit and observable.
- Per-request sequence lengths and block tables are passed to the kernels.
- There is no padded lane because requests are not fused into one kernel batch;
  each request's plan sees only its own table and positions.
- A request removal does not invalidate the KV blocks of other requests.
- Chunked prefill uses the same causal attention semantics as full prefill; the
  paged prefill kernel enforces the causal limit across chunk boundaries.
- Prefill and decode alternate rather than sharing a mixed-shape step. Fused
  mixed prefill/decode batching is out of scope.
- Prefill and decode are submitted to the same explicit model stream, so ordering
  is deterministic.

## 7. Paged KV cache

### 7.1 Configuration

- `F16` keys and values by default. `--kv-type f32` and `--kv-type q8_0` select
  opt-in `F32` and `Q8_0` pools for the `qwen35` and `qwen35moe` CPU and HIP
  backends. The prefix-cache model key includes the format, and the `F32` path
  remains the numerical baseline.
- HIP paged prefill attention selects the FP16 MMA kernel, a long-prefix
  split-key form, or the tiled fallback according to [kernels.md](kernels.md);
  all forms read the same paged KV layout.
- Configurable block size; the server and benchmark default to 16 tokens.
- One allocator per model with a fixed block count derived from context and
  concurrency, or set explicitly.
- Per-request block table mapping logical positions to physical blocks.
- The model and kernels address KV only through the block table; model code
  never sees physical block placement.

### 7.2 Components

| Component | Responsibility |
| --- | --- |
| **Block allocator** | Allocate, free, and reuse fixed-size blocks; track free capacity |
| **Request block table** | Map logical positions to physical blocks for each layer |
| **Reference counting** | Share blocks for prefix caching; free only at the last release |
| **Prefix cache** | Identify and reuse common prompt prefixes |
| **Cancellation cleanup** | Release all blocks owned by a cancelled or failed request |
| **Eviction** | Reclaim cached blocks under memory pressure |

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

- **Key:** a digest over a cache version tag, the model identity supplied by the
  server (by default the GGUF file SHA-256 plus the KV format), the block size,
  the token count, and every token. Lookup additionally compares the full token
  slice, so a hash collision cannot produce incorrect reuse.
- **Payload:** block references covering the prompt, the recurrent
  convolution/matrix state snapshot at the prompt boundary, and the logits for
  the last prompt token. The logits let a full-prompt hit sample immediately.
- **Partial final block:** when the prompt is not block-aligned, the cache owns
  the original partial block and the request receives a private copy, so later
  decode writes cannot corrupt the cached KV.
- **Eviction:** least-recently-used. The allocator invokes the cache when no
  free block remains; cache entries release their own references while
  request-held references keep shared blocks alive.
- **Insertion:** after a completed prompt, the serving manager splits the prompt
  prefix out of the request table and inserts it. Cache failure or allocation
  pressure never changes the request's generated output.

Hit rate and cache occupancy are reported by `/metrics`.

## 8. HIP Graph

The plain HIP stream path is the default and remains correct without graph
capture. Replay is opt-in and is introduced only after the decode host path is
measured.

Behavior:

- The decode plan is captured at its first eligible decode step and instantiated
  once per session.
- The captured graph addresses weights, activations, KV pools, the session block
  table, and the dynamic parameter buffer, all of which have stable addresses
  for the session lifetime.
- Kernel arguments do not change between replays. A small device buffer supplies
  the current token and position; it is uploaded on the same stream before the
  graph launch.
- Before replay the runtime verifies that the session is open, the position is
  inside capacity, the block table covers the next position, and the dynamic
  buffer is valid.
- If capture, parameter upload, or replay fails, the graph is destroyed and the
  same step is submitted through the plain plan. A capture failure disables
  graph use for that session; it never corrupts output.
- Prefill plans are not captured; the decode shape is the stable replay class.
- Graph resources are released with the session before the model.

Replay is not a default because the measured benefit depends on the context and
the host launch path; the model reports record the current measurements.

## 9. API surface

### 9.1 CLI

The CLI supports loading a GGUF model, single-prompt generation (including
stdin input), inspection, benchmark, and a diagnostic NPU-probe mode.
`yuuka run` and `yuuka bench` accept `--backend`, `--kv-type`,
`--speculative`, and the chunk and sampling controls. `--npu` records the NPU
capability result and falls back to the selected backend; it is diagnostic only,
and NPU support is a future task. Serving uses `yuuka-server`, which also
accepts `--kv-type` and `--npu`.

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
fields are rejected. `n != 1`, nonzero penalties, `logprobs: true`,
`response_format`, `tools`, and `stream_options` are rejected explicitly rather
than ignored. `logprobs: false` is accepted because it requests no logprobs.
`stop: null` and empty stop strings are rejected.

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
| `invalid_request` | 400 | Malformed JSON, trailing data, empty messages, invalid content, or rejected chat rendering |
| `unsupported_parameter` | 400 | A known OpenAI parameter outside the supported subset |
| `request_too_large` | 413 | Request body exceeds the 8 MiB limit |
| `context_length_exceeded` | 400 | Prompt alone exceeds the configured context |
| `model_not_found` | 404 | Requested model identifier is not loaded |
| `method_not_allowed` | 405 | An endpoint was called with the wrong HTTP method |
| `server_overloaded` | 429 | Queue depth limit reached; retryable |
| `insufficient_capacity` | 503 | KV pool cannot admit the request; retryable |
| `request_timeout` | 504 | Request deadline expired |
| `client_closed` | 499 | Client cancelled the request |
| `generation_error` | 500 | Model, device, or streaming failure |
| `tokenizer_error` | 500 | Tokenization failed while preparing the prompt; rejected chat rendering uses `invalid_request` |

## 10. Observability

The scheduler records:

- queue depth, active requests, last and maximum batch size, maximum active;
- queue wait, prefill, decode, and sampling totals;
- TTFT and TPOT percentiles and maximum over a bounded sample window. TTFT is
  measured from request arrival and includes queue wait and prefill; queue wait
  is also reported separately. Sampling time is reported separately and is not
  included in prefill or decode time;
- prompt and generated token totals;
- submissions, admissions, completions, failures, cancellations, timeouts,
  queue-full rejections, and capacity rejections.

The KV allocator reports total, used, free, shared, and block size. The prefix
cache reports entries, blocks, hits, misses, insertions, and evictions.

`/metrics` exposes these as `yuuka_*` Prometheus counters and gauges. Logging
includes the request ID, finish reason, token counts, phase timings, and prefix
hit; prompt contents are not logged.

## 11. Backpressure and failure handling

- Queue-depth and concurrency limits are enforced before task construction.
- Capacity is checked before KV allocation; allocation failure fails one
  request and releases its blocks and task.
- The model stream is explicit; a device error fails the active request and is
  surfaced as `generation_error`. The server keeps running while the device
  remains usable.
- On shutdown, HTTP stops accepting connections while the scheduler stops
  admission; active requests drain under one deadline before cancellation.
- A cancelled or failed request always releases its task, block table, and
  graph resources. The server never returns success for a failed generation.

## 12. Deferred: multiple models

Model unload and multi-model management are deferred. The server loads exactly
one model for its lifetime and uses the model identity in every prefix-cache key
so a future multi-model server cannot mix cache entries.
