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
- missing or non-Qwen chat templates.

Each rejection test asserts the error class and that the failed file is not left
mapped.

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

Plan submission is also tested: unsupported quantized dtypes, widths that
violate the block layout, batches above the measured limit, null buffers, and
undersized buffers must be rejected when the operation is added, not ignored at
execution time. Every dtype supported by the batched GEMV kernel is checked at
batch sizes 1, 2, 3, 4, 8, and 16.

Prefill operators have their own native tests in `tests/hip/prefill_test.cpp`:

- WMMA GEMM for `F32`, `F16`, `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, and
  `Q6_K` at `M` values 1, 7, 16, and 33 and non-tile-aligned `N`;
- the rocBLAS FP16 baseline against a float reference;
- row-batched and head-batched RMSNorm, batched RoPE, batch embedding, and
  batch KV writes;
- legacy single-token embedding for Q4_0 and Q5_0, compared against the same
  packed-row reference as the batch path (`tests/hip/embedding_test.cpp`);
- causal GQA Flash Attention with a non-zero chunk start position;
- the fused convolution, L2 normalization, gated delta-net recurrence, and
  gated RMSNorm chunk kernel at the model dimensions (128 rows, 16 groups,
  32 heads, state 128), including updated convolution and matrix state. The
  same operation is run twice from restored state and must be bitwise
  identical; this catches the shared-scratch and state-ordering races that a
  single execution can miss.

Each compares against an independent float implementation and records its
tolerance in the test.

Paged KV and graph replay have dedicated native coverage in
`tests/hip/paged_test.cpp`:

- KV writes through a permuted block table land in the mapped physical rows;
- paged decode and prefill attention match an independent float reference
  through the same table;
- device-to-device block copies used for private prefix blocks;
- graph capture and replay with device-resident token/position updates across
  several positions;
- rejection of plans containing position-dependent ops without a
  dynamic-parameter kernel, so the runtime can fall back instead of replaying
  stale arguments.

Serving tests are split by layer:

- `memory`: allocator reference counting, double-release rejection, table
  mapping and rollback, aligned and partial prefix splits with copy-on-write,
  longest-prefix lookup, exact-key comparison, LRU eviction, and allocation
  unblocking. These run with `-race`.
- `scheduler`: queue limits, admission rejection, queue and request timeouts,
  cancellation, build-failure slot release, decode-priority interleaving,
  explicit batch composition, shutdown drain, and latency summaries.
- `serving`: concurrent requests versus isolated runs, full and partial prefix
  reuse versus a fresh run, cancellation and injected decode-failure cleanup
  with balanced allocator counters, and repeated cycles with zero residual
  blocks. GPU session tests also assert that per-request device buffers do not
  grow the model-lifetime allocation list. Both CPU and HIP backends are covered
  when the HIP library is available.

### 3.4 End-to-end tests

- Fixed prompt, greedy sampling, fixed seed where applicable.
- Compare token IDs with the reference for a bounded number of tokens.
- Verify EOS and stop conditions.
- Verify chat template application and multi-turn context.
- Verify streaming deltas reconstruct the non-streaming response.
- Verify cancellation stops generation and releases resources.
- Verify concurrent requests do not affect each other's output.
- Verify prefix-cache reuse produces identical logits and tokens, including a
  longer prompt that reuses a shorter cached prefix.
- Verify a fully cached prompt samples from cached logits and matches a fresh
  run.
- Verify graph replay and fallback produce the same token sequence as a plain
  session, including an intentionally invalidated dynamic parameter buffer.
- Verify HTTP API tests remain independent of the real-model environment by
  using the tiny fixture for streaming, error, stop, and cancellation cases.
- Verify chunked prefill at chunk sizes that select batched GEMV and WMMA,
  including partial final chunks, matches unchunked CPU logits and the greedy
  top-1 token.
- Verify one decode step after chunked prefill matches the CPU state produced by
  a single unchunked prefill.
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
| Model | Qwen3.5-4B hybrid (primary), Qwen3 MoE F32/Q4_K fixtures, Qwen3 Dense, Llama Dense, small fixtures |
| Quantization | F16, Q8_0, Q6_K, Q5_K, Q4_K, Q5_0, Q4_0 |
| KV format | F32, F16, Q8_0 |
| Context | short, medium, maximum supported |
| Prompt language | English, Chinese, mixed |
| Batch | 1, small batch, continuous batch |
| Concurrency | 1, 2, 4 |
| Features | no cache, prefix cache, chunked prefill, graph replay, cancellation, speculative decoding including partial-accept rollback, grouped MoE experts |
| Backends | CPU reference, HIP GPU, library fallback |

NPU support is a future task and is not part of Phase 5 acceptance. The
`npu-probe` command remains a diagnostic check only.

### 5.2 Performance matrix

| Dimension | Values |
| --- | --- |
| Prefill length | 128 / 512 / 2048 / 8192 / 32768 tokens |
| Decode length | 128 / 256 tokens |
| Concurrency | 1 / 2 / 4 / 8 / 16 requests |
| Quantization | FP16, Q8_0, Q6_K, Q4_K, Q5_0, Q4_0 |
| Modes | latency and throughput |
| Comparison | llama.cpp HIP, llama.cpp Vulkan (same GGUF) |

Not every combination needs to run on every commit. For the first release, the
release gate is the recorded subset in [benchmarks.md](benchmarks.md):
prompt-length prefill curves at 128-32768 tokens and the 128/256-token decode
matrix at a 128-token prompt. The remaining concurrency and quantization rows
are tracked performance work for the features they cover; they do not block the
single-request GPU-only release. A smaller tracked subset runs regularly.

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
go test -tags hip ./...
go test -tags hip ./backend/hip ./engine/gpu ./serving ./cmd/yuuka-server
go test ./memory ./scheduler ./serving
ctest --test-dir build --output-on-failure
scripts/kernel_resources.sh
YUUKA_TEST_SPM_VOCAB=/path/to/ggml-vocab-llama-spm.gguf go test ./tokenizer
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
reference, layer-by-layer hidden-state comparison, the paged HIP executor, KV
formats, and speculative decoding against the plain greedy path. Synthetic
coverage includes F32 and Q4_K `qwen3moe` fixtures for chunked prefill and
prefill-plus-decode, plus a scripted partial-accept speculative rollback test.
HTTP API contract tests use the deterministic tiny fixture so they cover
streaming, errors, stops, and cancellation independently of the optional model
environment. SentencePiece golden IDs additionally accept
`YUUKA_TEST_SPM_VOCAB=/path/to/ggml-vocab-llama-spm.gguf`. Native GPU tests return CTest
skip code 77; Go GPU tests skip with a message when no HIP device is available
or `YUUKA_TEST_MODEL` is unset.

The Phase 0 hardware probe and the current CPU/HIP benchmark are run
explicitly:

```bash
./build/yuuka_hardware_probe
./build/yuuka_gemv_sweep
./build/yuuka_moe_sweep
go run ./cmd/yuuka npu-probe  # diagnostic only; NPU support is a future task
scripts/thermal_probe.sh 15 ./build/yuuka_hardware_probe
/tmp/yuuka-hip bench --model model.gguf --backend hip --prompt-tokens 16 --decode-tokens 16
/tmp/yuuka-hip bench --model model.gguf --backend hip --prompt-tokens 128 \
    --decode-tokens 32 --kv-type q8_0 --speculative
go build -tags hip -o /tmp/yuuka-serving-bench ./bench/serving
/tmp/yuuka-serving-bench --model model.gguf --backend hip --ctx 256 \
    --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16
/tmp/yuuka-serving-bench --model model.gguf --backend hip --ctx 256 \
    --concurrency 4 --requests 8 --prompt-tokens 64 --decode-tokens 16
```

The serving benchmark reports queue wait, TTFT, prefill, decode, and sampling
percentiles plus aggregate request and output-token throughput. Run one
benchmark process at a time; concurrent benchmark processes contend for the GPU
and invalidate both results.

A clean export of the tracked tree can be validated without relying on local
build artifacts:

```bash
scripts/check_clean_build.sh
scripts/check_clean_build.sh --hip
```

The first command checks formatting, vet, build, and race tests. The `--hip`
form also configures and builds the native library, runs CTest, and runs the
HIP-tagged Go tests in the exported tree.

## 10. CI and test execution

| Tier | Environment | Contents |
| --- | --- | --- |
| **Unit** | Any Linux machine | GGUF parser, tokenizer, metadata mapping, CPU reference, paged KV allocator/prefix cache, scheduler and serving state machines |
| **GPU correctness** | gfx1151 runner | Kernel tests, end-to-end generation, logits comparison |
| **GPU performance** | gfx1151 runner, controlled power profile | Tracked benchmark subset, nightly or per-release |
| **Compatibility** | gfx1151 runner | llama.cpp HIP/Vulkan comparison on the same GGUF |

Current status: the CPU tier is checked in as `.github/workflows/ci.yml` and
runs formatting, `go vet`, `go build`, and `go test -race` on every push and
pull request. The gfx1151 tier is `.github/workflows/hip.yml`; it is manual and
nightly because it requires a self-hosted ROCm runner with rocBLAS and rocWMMA.
The HIP workflow builds the native library, runs `ctest`, builds the `hip` Go
tags, and runs the HIP-tagged tests. Set the `YUUKA_TEST_MODEL` repository
secret to enable the real-model cases on that runner.

Tests that require the real model skip when `YUUKA_TEST_MODEL` is unset. A
separate subset of tests runs without a model file by generating deterministic
tiny GGUF fixtures from `tests/fixture`, including the tiny `qwen35` GPU/CPU
comparison, dense-model tests, and server request tests. Fixtures are generated
at test time and are not committed as model artifacts. Real-model coverage
remains opt-in.

Large model files are not committed to the repository. GPU tests must skip with
a clear message when no supported device is present; they must not silently
pass.

## 11. Milestone acceptance

Each roadmap phase has explicit acceptance criteria in
[roadmap.md](roadmap.md). This document defines how those criteria are measured:
the correctness layers above, the benchmark protocol, and the reporting
metadata. A phase is not complete because code merged; it is complete when its
acceptance evidence is recorded and reproducible.
