# Testing and Benchmarking

## 1. Principles

1. **Correctness before performance.** A fast kernel with incorrect logits is a
   regression, not an optimization.
2. **Reproducibility.** Every benchmark records the model file, quantization,
   ROCm and driver versions, sampling settings, warmup, and thermal conditions.
3. **Separate latency from throughput.** Single-request token/s and
   multi-request aggregate throughput are different metrics.
4. **Compare like with like.** Cross-framework comparisons use the same GGUF,
   prompt set, context length, sampling parameters, and hardware conditions.
5. **Test the seams.** The GGUF reader, tokenizer, planner, kernels, scheduler,
   and API each need independent tests in addition to end-to-end tests.

## 2. Correctness strategy

Correctness is established in layers, from format parsing to generated tokens:

```text
GGUF parsing and validation
        |
Tokenizer and chat template (token IDs)
        |
Tensor-level operator comparison vs CPU reference
        |
Logits comparison (tolerance)
        |
Greedy token sequence and stop behavior
        |
End-to-end chat and streaming behavior
```

A failure at any layer blocks the layers above it. Do not compensate for a
tokenizer mismatch by loosening logits tolerances.

### 2.1 Reference implementations

- **CPU reference:** straightforward, readable implementations prioritized for
  clarity, with documented accumulation precision. They are the primary oracle
  for tensor and operator tests.
- **Independent framework:** llama.cpp using the same GGUF is the primary
  external comparison for logits and token sequences. It is a reference, not an
  execution dependency.
- **Mathematical checks:** small hand-computed cases for normalization, RoPE,
  softmax, and quantization decode.

Reference code must not share the optimized kernel's implementation strategy;
otherwise a shared bug can pass both sides.

### 2.2 Tolerance policy

Tolerance is defined per stage, not only for final logits:

| Stage | Comparison | Policy |
| --- | --- | --- |
| Quantized weight decode | Reconstructed values vs exact reference | Exact for integer unpacking; documented tolerance for float scale application |
| GEMV / GEMM | Output tensor vs CPU reference | Absolute and relative tolerance per dtype |
| RMSNorm / RoPE | Output tensor vs CPU reference | Tight tolerance; test extreme values |
| Attention | Output vs CPU reference | Tolerance plus causal-mask and GQA-specific cases |
| Logits | Top-k values and ordering | Tolerance for values; exact top-1 for greedy tests |
| Tokens | Greedy sequence | Exact token IDs on the golden set |

The concrete numeric threshold for each stage is recorded with the test that
enforces it. It must be tight enough to catch a wrong quantization scale or a
missing mask, and loose enough to accommodate legitimate floating-point
reassociation. Per-operator tolerances do not bound network-level drift; a
route change must also clear the end-to-end chunk gate.

## 3. Test layers

### 3.1 GGUF reader tests

Checked-in rejection tests use synthetic and real fixtures to cover:

- valid single-file and split-file models;
- bad magic, unsupported version, truncated header, unknown metadata type;
- duplicate metadata keys and tensor names, zero dimensions, rank overflow;
- offsets that overlap, exceed the data section, or violate alignment;
- incomplete shards and tensors assigned to multiple shards;
- quantization block sizes that do not divide the tensor dimensions;
- missing required tensors, and inconsistent head/KV-head/head-dimension or RoPE
  section metadata;
- special-token ID range checks and non-Qwen chat-template rejection at render
  time.

Every rejection test asserts the error class and that the failed file is not
left mapped.

### 3.2 Tokenizer tests

Token IDs are the contract. Golden tests must cover:

- English and Chinese text;
- whitespace, newlines, punctuation, and emoji;
- special tokens and role-marker fragments;
- BOS/EOS insertion and suppression.

Multi-turn chat rendering and stop-string behavior are checked at the rendered
text level. Every golden token-ID case compares against hardcoded reference IDs,
and decoded text is checked as a secondary test.

### 3.3 Operator tests

For every operator and quantization type:

- random shapes, including non-aligned `M`, `N`, and `K`;
- small hand-checkable shapes;
- boundary values and extreme magnitudes;
- tensors with a single row or column;
- NaN/Inf behavior where defined;
- deterministic comparisons;
- CPU reference, GPU kernel, and library fallback where available.

Tests must cover the actual packed layouts used by GGUF, not only dense FP16
inputs.

Plan submission is also tested: unsupported dtypes, widths that violate the
block layout, batches above the measured limit, null buffers, and undersized
buffers must be rejected when the operation is added, not ignored at execution
time.

Each stage compares against an independent implementation and records its
tolerance in the test.

### 3.4 Model and execution tests

Real-model tests are opt-in and cover:

- GGUF loading and tokenizer golden IDs;
- greedy golden token sequences and determinism;
- logits tolerance against a llama.cpp CPU reference;
- layer-by-layer hidden-state comparison between CPU and HIP;
- the paged HIP executor, the KV formats, and long-context split-key attention;
- speculative decoding against the plain greedy path;
- chunked prefill, including partial final chunks and decode after prefill;
- the `qwen35moe` CPU reference and HIP decoder, including the routed and shared
  expert paths and the IQ codebooks;
- the tiny fixtures on both CPU and HIP paths, which run without a model file.

Real-model chunked prefill is gated on behavior: the CPU top-1 token and a
documented logit bound, not a tight logit equality. The device prefill path and
the CPU reference use different arithmetic, and the MoE router is sensitive to
hidden-state drift.

### 3.5 Serving tests

Serving tests are split by layer:

- **memory:** allocator reference counting, double-release rejection, table
  mapping and rollback, aligned and partial prefix splits with copy-on-write,
  longest-prefix lookup, exact-key comparison, LRU eviction, and allocation
  unblocking. These run with `-race`.
- **scheduler:** queue limits, admission rejection, queue and request timeouts,
  cancellation, build-failure slot release, decode-priority interleaving,
  explicit batch composition, shutdown drain, and latency summaries.
- **serving:** concurrent requests versus isolated runs, full and partial prefix
  reuse versus a fresh run, cancellation and injected decode-failure cleanup
  with balanced allocator counters, and repeated cycles with zero residual
  blocks.
- **HTTP API:** streaming, errors, stop handling, cancellation, and request
  limits, using the deterministic tiny fixture so the tests do not depend on a
  real model.

### 3.6 End-to-end tests

- Fixed prompt, greedy sampling, fixed seed where applicable.
- Compare token IDs with the reference for a bounded number of tokens.
- Verify EOS and stop conditions.
- Verify chat template application and multi-turn context.
- Verify streaming deltas reconstruct the non-streaming response.
- Verify cancellation stops generation and releases resources.
- Verify concurrent requests do not affect each other's output.
- Verify prefix-cache reuse produces identical tokens and text, including a
  longer prompt that reuses a shorter cached prefix.
- Verify a fully cached prompt takes the cached-prefix path.
- Verify graph replay and fallback produce the same token sequence as a plain
  session, including an intentionally invalidated dynamic parameter buffer.
- Verify chunked prefill matches unchunked CPU logits and the greedy top-1
  token, and that one decode step after chunked prefill matches the CPU state
  produced by a single unchunked prefill.
- Verify invalid and oversized chunk sizes are rejected or clamped explicitly.

## 4. Cross-framework comparison

For llama.cpp comparisons:

- use the identical GGUF file and quantization;
- use the same prompt tokens, context length, and sampling parameters;
- compare logits at selected positions with a documented tolerance;
- compare greedy token sequences;
- run on the same device, power profile, and ROCm/driver version;
- record the llama.cpp build/commit and backend (HIP or Vulkan).

Differences in tokenizer or chat template must be resolved before comparing
model output. A text-only comparison is insufficient.

## 5. Test matrix

### 5.1 Correctness matrix

| Dimension | Values |
| --- | --- |
| Model | Qwen3.5-4B hybrid (primary), Qwen3.6-35B-A3B MoE (CPU and HIP), tiny `qwen35` and `qwen35moe` fixtures |
| Quantization | F16, Q8_0, Q6_K, Q5_K, Q4_K, Q5_0, Q4_0, IQ2_XXS, IQ2_S, IQ3_S |
| KV format | F32, F16, Q8_0 |
| Context | short, medium, maximum supported |
| Prompt language | English, Chinese, mixed |
| Batch | 1, small batch, continuous batch |
| Concurrency | 1, 2, 4 |
| Features | no cache, prefix cache, chunked prefill, graph replay, cancellation, speculative decoding including partial-accept rollback, routed and shared MoE experts, selected and grouped MoE kernels, IQ codebook experts |
| Backends | CPU reference, HIP GPU, library fallback |

`qwen35moe` runs on the CPU reference and the HIP executor; the HIP planner
rejects an expert dtype the MoE kernels cannot decode and any expert tensor
whose row width violates the block layout. Architectures other than `qwen35` and
`qwen35moe` are rejected at load time and have no test matrix entry.

NPU support is a future task and is not part of acceptance. The `npu-probe`
command remains a diagnostic check only.

### 5.2 Performance matrix

| Dimension | Values |
| --- | --- |
| Prefill length | 128 / 512 / 2048 / 8192 / 32768 tokens |
| Decode length | 128 / 256 tokens |
| Concurrency | 1 / 2 / 4 / 8 / 16 requests |
| Quantization | FP16, Q8_0, Q6_K, Q4_K, Q5_0, Q4_0 |
| Modes | latency and throughput |
| Comparison | llama.cpp HIP, llama.cpp Vulkan (same GGUF) |

Not every combination needs to run on every commit. The release gate is the
subset recorded in the model reports indexed by [benchmarks.md](benchmarks.md);
the remaining concurrency and quantization rows are tracked performance work for
the features they cover. A smaller tracked subset runs regularly. The longest
verified prompt and every unmeasured row are listed in each report's Known
limits.

## 6. Benchmark protocol

### 6.1 Required metadata

The required report metadata, including the model file name and hash,
environment, versions, build, and protocol fields, is defined in
[benchmarks.md](benchmarks.md#model-identity-and-environment). Every benchmark
run records those fields; a result without them is an anecdote, not a result.

### 6.2 Procedure

1. Start from a clean process and load the model.
2. Record load time and initial memory.
3. Run warmup iterations and discard them from performance results.
4. Measure steady-state prefill and decode separately.
5. Repeat each configuration enough times to report P50/P95/P99, not a single
   run.
6. Interleave configurations where practical to reduce thermal and clock bias.
7. Record clock, power, and throttling indicators if available.
8. Stop on thermal or power throttling and mark the result accordingly.

Do not compare results collected under different power profiles, clock states,
or background load without labeling the difference.

### 6.3 Test inputs must expose the failure

Choose test inputs that make the mistake under test visible, and say so in the
test. A kernel that applies a gate with the wrong exponent can agree with a
correct kernel at one input and differ by more than the tolerance at another.
The rules:

- For every decay, gate, scale, or mask the kernel applies, pick at least one
  test value where applying it and not applying it differ by far more than the
  tolerance.
- Compare against an independent implementation, not only against the path being
  replaced: the two device paths can share a mistake, and one of them is usually
  the reference the other is compared against.
- State the input choice in the test so the reason survives the next edit.

### 6.4 What to measure separately

- model load and first-use warmup;
- tokenizer and prompt preparation;
- queue wait and admission;
- pure prefill GPU time;
- pure decode GPU time;
- sampling and detokenization;
- kernel launch and host scheduling overhead;
- KV-cache read cost;
- end-to-end request latency;
- model size and batch-size effects.

Isolated kernel benchmarks are necessary but not sufficient. A kernel that wins
in isolation can lose end-to-end because of launch overhead or occupancy.

Per-operation attribution is a diagnostic step, not a benchmark result. Use it
to rank work, then confirm any accepted change with the protocol above and
record it in the model's benchmark report. Attribution instrumentation adds a
submission gap per operation, so its totals exceed a release run.

## 7. Metric definitions

| Metric | Definition |
| --- | --- |
| **TTFT** | Time from request arrival to the first streamed token, including queueing and prefill. Report queue wait separately if possible. |
| **TPOT** | Mean time per output token after the first token, measured over the decode phase. |
| **Inter-token latency** | Distribution of individual gaps between output tokens; report percentiles. |
| **Prefill tok/s** | Prompt tokens divided by prefill duration, excluding queue wait. |
| **Decode tok/s** | Generated tokens divided by decode duration. For batches, report aggregate and per-request values. |
| **Aggregate throughput** | Total output tokens per second across concurrent requests. |
| **Effective weight bandwidth** | Weight bytes read per second implied by decode throughput; label it as an effective estimate, not a hardware counter. |
| **Peak RAM / device memory** | Maximum resident memory during the run, including KV and workspace. |
| **Occupancy / LDS / VGPR** | Profiler-reported kernel resource usage, recorded per tracked kernel. |

## 8. Reporting template

The report layout, section order, and per-table fields are defined in
[benchmarks.md](benchmarks.md#report-structure). Every model report records the
model identity block and the protocol row for each table. A benchmark result
without that metadata is an anecdote, not a result.

## 9. Regression gates

### 9.1 Correctness gates

- Golden token sequences must match exactly for greedy decoding.
- Logits must remain within the documented tolerance.
- Tokenizer golden tests must pass exactly.
- No test may be skipped or loosened to accommodate a kernel change without an
  explicit review decision.

### 9.2 Resource gates

- Repeated request cycles must not grow KV or scratch usage without bound.
- Cancellation and error paths must return all blocks and buffers.
- Graph replay must not retain per-request resources after completion.
- A model reload must not leak device memory.

### 9.3 Performance gates

Performance gates are defined after the first stable baseline, using measured
run-to-run variance. A change is blocked if it causes a statistically
significant regression on a tracked shape outside the accepted noise band.
Thresholds must be recorded with the baseline; do not invent a universal
percentage before measurement.

## 10. CI and test execution

| Tier | Environment | Contents |
| --- | --- | --- |
| **Unit** | Any Linux machine | GGUF parser, tokenizer, metadata mapping, CPU reference, paged KV allocator/prefix cache, scheduler and serving state machines |
| **GPU correctness** | gfx1151 runner | Kernel tests, end-to-end generation, logits comparison |
| **GPU performance** | gfx1151 runner, controlled power profile | Tracked benchmark subset, nightly or per-release |
| **Compatibility** | gfx1151 runner | llama.cpp HIP/Vulkan comparison on the same GGUF |

The canonical build, test, and formatting commands are defined in
[AGENTS.md](../AGENTS.md) section 11 and must stay working. The `hip` build tag
is required for the HIP executor and its Go binding; the default build does not
require the native shared library.

Tests that require a real model or a GPU are opt-in. They skip with a clear
message when the prerequisite is absent and must never silently pass. Large
model files are not committed to the repository; deterministic tiny GGUF
fixtures are generated at test time.

## 11. Milestone acceptance

This document defines how phase acceptance is measured: the correctness layers
above, the benchmark protocol, and the reporting metadata. A phase is not
complete because code merged; it is complete when its acceptance evidence is
recorded and reproducible. The phase gates are listed in
[AGENTS.md](../AGENTS.md) section 8.
