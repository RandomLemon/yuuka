# Serving and Scheduling

## 1. Scope

This document covers the request path above the model executor: admission,
batching, paged KV cache, HIP Graph replay, APIs, and observability. It assumes
the single-request GPU path is already correct and measurable.

The first release prioritizes a single model and a single request. Continuous
batching, prefix caching, and graph replay are introduced only after the
single-request baseline is stable.

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

## 3. Request lifecycle

| State | Meaning | Exit conditions |
| --- | --- | --- |
| **Queued** | Accepted but not yet admitted | Admitted, rejected, timed out, client cancelled |
| **Prefill** | Prompt chunks are being processed | Prefill complete, cancelled, failed |
| **Decode** | Tokens are being generated | Stop condition, max tokens, cancelled, failed |
| **Finishing** | Sampling and stream flush | Response completed |
| **Closed** | Resources released | Terminal state |

Each request carries a stable ID used by logs, metrics, cancellation, and
cleanup. A request that fails must release its KV blocks, scratch, and stream
state exactly once.

## 4. Admission and budgets

Admission prevents overcommit and keeps latency predictable. It accounts for:

- **Token budget:** prompt plus reserved output tokens, bounded by the model
  context length.
- **KV budget:** blocks required for the prompt and the maximum generated
  sequence, including prefix-cache sharing.
- **Concurrency limit:** maximum active requests and maximum batch size.
- **Scratch and graph workspace:** plan-specific temporary memory.

Initial policy:

- Reject a request that cannot fit even with an empty queue, returning a clear
  capacity error.
- Otherwise queue it, subject to a queue-depth limit and request timeout.
- Reserve KV capacity for the maximum output length when configured; otherwise
  reserve incrementally and fail the request if it exhausts its budget.
- Never admit a request by silently truncating its context. Truncation, if
  supported, is an explicit, documented request option.

Exact queueing and rejection semantics are configuration, not hard-coded
behavior. The implementation must expose the chosen limits and report them in
health and metrics endpoints.

## 5. Scheduling modes

| Mode | Goal | Batch policy |
| --- | --- | --- |
| **Latency** | Minimize TTFT and TPOT for one user | Small batches; prioritize the active request; avoid unnecessary prefill interleaving |
| **Throughput** | Maximize aggregate tokens/s | Continuous batching; reuse weights across requests; interleave prefill and decode within budget |

The mode is a server-level or request-class configuration. It must not change
model semantics or numerical results.

### 5.1 Initial scheduling policy

- Decode steps have priority over new prefill work to protect TPOT.
- Prefill is split into chunks; a long prompt cannot occupy the GPU for an
  unbounded period while other requests are decoding.
- A newly admitted request receives a bounded prefill opportunity, then yields
  to pending decode work.
- The scheduler records why a request was delayed: queueing, prefill chunking,
  KV pressure, or GPU execution.

The precise policy is expected to be tuned. The policy must be observable and
must not starve a request indefinitely.

## 6. Continuous batching and chunked prefill

Continuous batching admits new requests between decode steps rather than waiting
for the entire batch to finish. The batch is recomposed as requests arrive,
complete, or cancel.

Requirements:

- Batch composition is explicit and recorded for each step.
- Per-request sequence lengths and block tables are passed to the kernels.
- Padding is minimized; padded lanes must not affect sampling or KV writes.
- A request removal must not invalidate the KV blocks of other requests.
- Chunked prefill uses the same attention semantics as full prefill, including
  causal masking across chunk boundaries.
- Prefill and decode may share a step only if the executor and kernels support
  the mixed shape; otherwise they alternate under the scheduler's budget.

The initial target is correctness and reproducible throughput, not a specific
batching algorithm. Batch-size and chunk-size thresholds are measured.

## 7. Paged KV cache

### 7.1 Initial configuration

- FP16 keys and values.
- Configurable block size, initially 16 or 32 tokens.
- Per-request block table mapping logical positions to physical blocks.
- A logical KV view that does not expose physical block placement to model code.

Quantized KV, sliding windows, and hybrid-attention state caches are extension
points, not first-release requirements.

### 7.2 Components

| Component | Responsibility |
| --- | --- |
| **Block allocator** | Allocate, free, and reuse fixed-size KV blocks; track free capacity |
| **Request block table** | Map logical token positions to physical blocks for each layer/head |
| **Reference counting** | Share blocks for prefix caching; free only when the last owner releases |
| **Prefix cache** | Identify and reuse common prompt prefixes |
| **Cancellation cleanup** | Release all blocks owned by a cancelled or failed request |
| **Eviction** | Reclaim cached blocks under memory pressure |

### 7.3 Invariants

- A block is freed exactly once.
- A block is not returned to the free list while any request or cache entry
  still references it.
- Kernel reads never cross a block boundary unless the kernel explicitly handles
  the block table.
- The block table is immutable for the duration of a submitted step, or updated
  only at a documented synchronization point.
- KV writes for a token occur before any later decode step reads that token.
- A request cannot observe blocks belonging to another request.

### 7.4 Prefix cache

A prefix-cache entry represents one or more complete token blocks shared by
multiple requests. The cache key must include the model identity and all
tokenizer/model settings that affect the KV values. Hash collisions must not
produce incorrect reuse; a full key comparison or a collision-resistant key is
required.

Prefix entries are reference-counted and evicted under a documented policy.
Eviction policy is an open decision: candidates include LRU and cost-aware
eviction based on block count and reuse probability. The first implementation
should choose one simple policy, measure hit rate, and record the tradeoff.

## 8. HIP Graph

The plain HIP stream path is the default and must remain correct without graph
capture. Graph replay is introduced only when profiling shows that host launch
overhead is a significant part of decode time.

Graph cache keys include:

- batch size and shape class;
- model and quantization type;
- kernel sequence and graph topology;
- whether KV block tables and sequence lengths are updated in place.

Requirements:

- Stable device addresses for weights, scratch, and graph workspace.
- A documented mechanism to update block-table pointers, sequence lengths, and
  other dynamic parameters.
- Validation before replay; if a parameter cannot be updated safely, fall back
  to normal submission.
- No graph replay for a request that has been cancelled or whose buffers have
  been released.
- Graph resources are released with the executable or model.

Graph capture is not a substitute for a correct stream path. A graph failure
must degrade to normal execution, not corrupt output.

## 9. API surface

### 9.1 CLI

The CLI should support:

- loading a GGUF model and reporting its configuration and quantization;
- interactive prompt and generation;
- a performance report with prefill and decode timings;
- a benchmark mode with fixed prompt lengths and decode counts.

Illustrative commands:

```text
yuuka run --model model.gguf --ctx 8192
yuuka bench --model model.gguf --prompt-tokens 2048 --decode-tokens 128
yuuka inspect --model model.gguf
```

### 9.2 HTTP

The server targets an OpenAI-compatible subset:

| Surface | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Chat generation, streaming or non-streaming |
| `GET /v1/models` | Loaded model metadata and capabilities |
| `GET /healthz` | Liveness and device readiness |
| `GET /metrics` | Runtime metrics in a machine-readable format |

Streaming uses server-sent events with incremental deltas and a terminal
completion event. Unsupported OpenAI parameters must be rejected explicitly or
documented as ignored; they must not silently change semantics.

Request parameters include context length, temperature, top-p, top-k, seed,
stop sequences, maximum output tokens, and concurrency class where supported.
Sampling behavior must be identical between CLI and HTTP for the same
parameters and seed.

### 9.3 Error model

API errors should be structured and include:

- a stable error code;
- a human-readable message;
- the request ID;
- whether the request can be retried;
- capacity details for admission failures.

A cancelled client connection must propagate cancellation to the scheduler and
release resources. It must not leave a request decoding in the background.

## 10. Observability

| Metric | Why it matters |
| --- | --- |
| Queue depth and wait time | Admission and fairness |
| Active requests and batch size | Batching efficiency |
| TTFT | Prefill and queueing latency |
| TPOT / inter-token latency | Decode smoothness |
| Prefill and decode tokens/s | Throughput split |
| KV blocks used / free / cached | Memory pressure and prefix-cache value |
| Prefix-cache hit rate | Cache effectiveness |
| Kernel time and launch overhead | Where decode time goes |
| Sampling time | Host-side overhead |
| Cancellation and error counts | Correctness of lifecycle |
| Peak RAM / device memory | Capacity planning |

Metrics must distinguish single-request latency from aggregate throughput.
Percentiles (P50/P95/P99) are required for latency reporting; averages alone are
insufficient.

Logging should include the request ID, scheduler decisions, prefill chunk
boundaries, decode step counts, cancellation, and terminal status. Logs must not
include prompt contents by default.

## 11. Backpressure and failure handling

- Enforce queue-depth and concurrency limits.
- Reject work before KV allocation when capacity is insufficient.
- Reserve headroom for scratch and graph workspace; do not fill device memory
  to the last block.
- On device allocation failure during a request, fail that request, release its
  resources, and keep the server alive if the device remains usable.
- On device loss, fail all active requests and report an unhealthy state.
- On shutdown, stop admission, cancel or drain active requests according to
  configuration, and release model resources.

The server must never silently drop a request or return a successful response
for a generation that failed.

## 12. Deferred: multiple models

Model unload and multi-model management are deferred until the serving path is
stable. The first release keeps one model resident. The model manager interface
should not preclude later support, but it must not add complexity to the
single-model hot path.
