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

| Stage | Comparison | Suggested policy |
| --- | --- | --- |
| Quantized weight decode | Reconstructed values vs exact reference | Exact for integer unpacking; documented tolerance for float scale application |
| GEMV / GEMM | Output tensor vs CPU reference | Absolute and relative tolerance per dtype |
| RMSNorm / RoPE | Output tensor vs CPU reference | Tight tolerance; test extreme values |
| Attention | Output vs CPU reference | Tolerance plus causal-mask and GQA-specific cases |
| Logits | Top-k values and ordering | Tolerance for values; exact top-1 for greedy tests |
| Tokens | Greedy sequence | Exact token IDs on the golden set |

The concrete numeric thresholds are recorded per test and per dtype. They must
be tight enough to catch a wrong quantization scale or a missing mask, and loose
enough to accommodate legitimate floating-point reassociation.

## 3. Test layers

### 3.1 GGUF reader tests

Use synthetic and real fixtures to cover:

- valid single-file and split-file models;
- bad magic, unsupported version, truncated header;
- malformed metadata types and duplicate keys;
- duplicate tensor names, zero dimensions, rank overflow;
- offsets that overlap, exceed the data section, or violate alignment;
- incomplete shards and tensors assigned to multiple shards;
- quantization block sizes that do not divide the tensor dimensions;
- missing required tensors and unexpected extra tensors;
- metadata inconsistency (heads, KV heads, head dimension, context length);
- tokenizer metadata with invalid special token IDs;
- chat templates that fail to parse.

Each rejection test asserts the error class and that no device allocation or
mapping leak occurred.

### 3.2 Tokenizer tests

Token IDs are the contract. Golden tests must include:

- English and Chinese text;
- whitespace, newlines, tabs, and punctuation;
- Unicode normalization edge cases and emoji;
- special tokens and role markers;
- BOS/EOS insertion and suppression;
- multi-turn chat template rendering;
- stop strings and maximum-token termination;
- empty prompt and single-character prompt.

Every golden case compares token IDs against the reference implementation, and
decoded text as a secondary check.

### 3.3 Operator tests

For every operator and quantization type:

- random shapes, including non-aligned `M`, `N`, and `K`;
- small hand-checkable shapes;
- boundary values and extreme magnitudes;
- tensors with a single row or column;
- NaN/Inf behavior where defined;
- deterministic mode comparisons;
- CPU reference, GPU kernel, and library fallback where available.

Tests must cover the actual packed layouts used by GGUF, not only dense FP16
inputs.

### 3.4 End-to-end tests

- Fixed prompt, greedy sampling, fixed seed where applicable.
- Compare token IDs with the reference for a bounded number of tokens.
- Verify EOS and stop conditions.
- Verify chat template application and multi-turn context.
- Verify streaming deltas reconstruct the non-streaming response.
- Verify cancellation stops generation and releases resources.
- Verify concurrent requests do not affect each other's output.
- Verify prefix-cache reuse produces identical logits and tokens.

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
| Model | Qwen3.5-4B hybrid (primary), Qwen3 Dense, Llama Dense, small fixtures |
| Quantization | F16, Q8_0, Q6_K, Q4_K |
| Context | short, medium, maximum supported |
| Prompt language | English, Chinese, mixed |
| Batch | 1, small batch, continuous batch |
| Concurrency | 1, 2, 4 |
| Features | no cache, prefix cache, chunked prefill, graph replay, cancellation |
| Backends | CPU reference, HIP GPU, library fallback |

### 5.2 Performance matrix

| Dimension | Values |
| --- | --- |
| Prefill length | 128 / 512 / 2048 / 8192 / 32768 tokens |
| Decode length | 128 / 256 tokens |
| Concurrency | 1 / 2 / 4 / 8 / 16 requests |
| Quantization | FP16, Q8_0, Q6_K, Q4_K |
| Modes | latency and throughput |
| Comparison | llama.cpp HIP, llama.cpp Vulkan (same GGUF) |

Not every combination needs to run on every commit. The full matrix is a
release and milestone gate; a smaller tracked subset runs regularly.

## 6. Benchmark protocol

### 6.1 Required metadata

Record at minimum:

- model name, file hash, and quantization preset;
- per-tensor dtype summary, if the preset is mixed;
- tokenizer and chat-template version;
- prompt set and token lengths;
- ROCm version, HIP runtime, kernel/driver version, and firmware;
- gfx target and device name;
- host CPU, memory configuration, and power/thermal profile;
- code commit, kernel version, and dispatch-table version;
- sampling parameters and seed;
- warmup iterations and measurement repetitions;
- concurrent request schedule and arrival pattern.

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

### 6.3 What to measure separately

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

| Field | Value |
| --- | --- |
| Date / commit | |
| Model / hash | |
| Quantization | |
| Backend / ROCm / driver | |
| Device / power profile | |
| Prompt tokens / output tokens | |
| Concurrency | |
| TTFT P50 / P95 / P99 | |
| TPOT P50 / P95 / P99 | |
| Prefill tok/s | |
| Decode tok/s (per request / aggregate) | |
| Effective weight GB/s | |
| Peak memory | |
| Correctness status | |
| Notes / throttling | |

A benchmark result without this metadata is an anecdote, not a result.

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

## 9.4 Current test commands

The repository uses these commands:

```bash
go build ./...
go test ./...
go test -race ./...
go build -tags hip ./...
go test -tags hip ./backend/hip ./engine/gpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

The `hip` build tag is required for the HIP executor and its Go binding. The
default build does not require the native shared library.

Real-model tests are opt-in and skip when the variable is absent:

```bash
YUUKA_TEST_MODEL=/path/to/model.gguf go test ./...
```

The real-model suite covers GGUF loading, tokenizer golden IDs, greedy golden
token sequences, determinism, logits tolerance against a llama.cpp CPU
reference, and server request/stream behavior. GPU tests return the CTest skip
code when no HIP device is available.

The Phase 0 hardware probe and the current CPU/HIP benchmark are run
explicitly:

```bash
./build/yuuka_hardware_probe
/tmp/yuuka-hip bench --model model.gguf --backend hip --prompt-tokens 16 --decode-tokens 16
```

## 10. CI and test execution

| Tier | Environment | Contents |
| --- | --- | --- |
| **Unit** | Any Linux machine | GGUF parser, tokenizer, metadata mapping, CPU reference, scheduler logic |
| **GPU correctness** | gfx1151 runner | Kernel tests, end-to-end generation, logits comparison |
| **GPU performance** | gfx1151 runner, controlled power profile | Tracked benchmark subset, nightly or per-release |
| **Compatibility** | gfx1151 runner | llama.cpp HIP/Vulkan comparison on the same GGUF |

Large model files are not committed to the repository. Fixtures are downloaded
by hash or generated synthetically. GPU tests must skip with a clear message
when no supported device is present; they must not silently pass.

## 11. Milestone acceptance

Each roadmap phase has explicit acceptance criteria in
[roadmap.md](roadmap.md). This document defines how those criteria are measured:
the correctness layers above, the benchmark protocol, and the reporting
metadata. A phase is not complete because code merged; it is complete when its
acceptance evidence is recorded and reproducible.
