# Roadmap

## 1. Strategy

The first executable vertical slice is:

> **Linux + ROCm + gfx1151 + Qwen3.5 hybrid + Q4_K GGUF + single-request,
> GPU-only inference.**

The sequence is deliberate:

1. Establish a reliable, measurable correctness baseline.
2. Optimize the decode hot path with native quantized GEMV.
3. Optimize prefill with WMMA and Flash Attention.
4. Add concurrent serving, paged KV, and graph replay.
5. Expand architecture, quantization, and NPU coverage.

Each phase has an evidence-based exit gate. A phase is not complete because code
has merged; it is complete when its acceptance results are reproducible under
the protocol in [testing.md](testing.md). No calendar dates are assigned here;
duration depends on hardware availability and measured progress.

## 1.1 Current implementation status

The Phase 0 toolchain is proven and recorded in [hardware.md](hardware.md).
The CPU reference and HIP single-request vertical slices for `qwen35` are
implemented with a CLI and server. HIP allocation, transfers, streams, events,
planned operators, and the full decoder execution path are implemented and
tested; CPU and HIP greedy tokens match.

Phase 0 exit criteria are met: the hardware report records the toolchain,
firmware, power-management profile, short sustained-load telemetry, and a clean
checkout is verified by `scripts/check_clean_build.sh` plus CI. Phase 1 exit criteria are met for the validated
models: CPU reference paths exist for `qwen35`, dense Llama, and dense Qwen3,
the Qwen and Llama tokenizers match reference token IDs, CPU/GPU logits and
layer traces agree, and a validated F32/F16 GEMM ABI baseline is checked in.
The decode path's first-step maximum absolute logit difference is `4.58e-5`; the
chunked prefill path reports `3.65e-3` on the same prompt with the same top-1
token, and chunk-size equivalence at 1024 tokens is exact.
HIP execution for the dense models is not implemented.

Phase 2 exit criteria are met: quantized GEMV is correct across randomized
shapes, GEMV accounts for about 90.5% of profiled decode time, and
per-quantization and per-batch baselines are recorded.

Phase 3 exit criteria are met: prompt-length curves at 128, 512, 2048, 8192, and
32768 tokens are recorded in [benchmarks.md](benchmarks.md), together with
same-GGUF comparisons against llama.cpp HIP and Vulkan. The measured
`SelectPrefillPath` table selects the custom WMMA GEMM or the rocBLAS FP16
baseline only in shape classes where the sweep showed them faster than the
compared baseline. Sixteen-token prefill latency improved from 1.433 s to
0.137 s and decode is unchanged. Chunked prefill passes logit/state equivalence
tests at chunk sizes 1-64, produces bitwise-identical logits for a 1024-token
prompt at chunk sizes 128, 512, and 1024, and completes a 32768-token prompt.
VGPR, LDS, scratch, and occupancy are recorded. The prompt path remains 5-16x
behind llama.cpp HIP and 7-18x behind llama.cpp Vulkan across the measured
prompt lengths; the remaining gap has not been fully attributed, and the 32-row
M tile plus scalar attention are follow-up optimizations rather than exit
criteria.

Phase 4 exit criteria are met: paged KV with a block allocator and immutable
per-request block tables, continuous scheduling with explicit prefill/decode
interleaving, a collision-safe LRU prefix cache that also snapshots recurrent
state, opt-in HIP Graph capture/replay with fallback, an OpenAI-compatible SSE
API, health/metrics/pprof endpoints, cancellation and timeout handling, and
resource-reclamation tests are implemented. Concurrent CPU and HIP requests
reproduce isolated-run tokens, repeated request cycles leave no KV blocks
allocated, and cancellation releases every block. Aggregate decode throughput
does not scale with concurrency because mixed-shape batched kernels are not
implemented; requests are interleaved on one stream.

Phase 5 features are implemented and opt-in: `qwen3moe` expert routing with
grouped HIP expert kernels, F16 and Q8_0 KV cache formats, exact greedy
prompt-lookup speculative decoding, Q4_0/Q5_0 tensor support, and an NPU
capability probe with explicit GPU fallback. Each feature is covered by CPU
reference tests and a measured result in [benchmarks.md](benchmarks.md). NPU
execution itself remains disabled because the target host has the device and
`amdxdna` driver but no XRT userspace runtime; `--npu` therefore records the
capability result and falls back to the selected backend instead of claiming
acceleration.

## 2. Phase overview

| Phase | Objective | Key deliverables | Exit gate |
| --- | --- | --- | --- |
| **0. Bootstrap** | Prove the toolchain and characterize the device | Hardware report, repository scaffold, C ABI, GGUF fixtures, CPU reference skeleton | HIP/rocBLAS/WMMA smoke tests and allocation/bandwidth measurements recorded |
| **1. Minimal loop** | Correct single-request generation | GGUF reader, tokenizer, Qwen3/Llama dense, CPU reference ops, HIP FP16 baseline, CLI | Fixed-prompt token IDs match or logits are within documented tolerance; generation stops correctly |
| **2. Quantized decode** | Native low-bit decode | Q4_K/Q6_K/Q8_0 GEMV, fused unpack/scale, dispatch thresholds, profiling | Qwen3.5-4B Q4_K_M runs correctly; decode bottleneck is attributable to a kernel and memory behavior |
| **3. High-performance prefill** | Fast prompt processing | WMMA quantized GEMM, Flash Attention, chunked prefill, key fusions, tuning tables | Performance curves at 128/512/2048/8192/32768 tokens and same-condition comparison with llama.cpp |
| **4. Multi-request serving** | Concurrent, cache-aware service | Paged KV, continuous batching, prefix cache, HIP Graph, SSE/OpenAI API, metrics | Reproducible latency and throughput; cancellation and reclamation are leak-free |
| **5. Expansion** | Broader models and research | MoE/grouped GEMM, KV quantization, speculative decoding, NPU experiments, more architectures | Each feature is opt-in, measured, and does not regress the GPU-only path |

## 3. Phase 0 — Bootstrap and Hardware Validation

### Objective

Validate the actual gfx1151 software and memory environment before committing to
kernel designs or performance assumptions.

### Deliverables

- Device report: GPU visibility, ROCm/HIP version, kernel and firmware, available
  CUs, clocks, and power/thermal behavior.
- Library validation: rocBLAS GEMM, rocWMMA availability, and a minimal custom
  HIP kernel.
- Memory matrix: measured bandwidth, latency, and synchronization cost for
  `hipMalloc`, host pinned memory, mapped memory, and any unified-memory path.
- Repository scaffold: Go module, CMake HIP shared library, stable C ABI,
  CLI/server skeletons, test harness, and CI.
- Small dense model fixture and a tokenizer/logits comparison harness.
- CPU reference skeletons for the operations needed by Phase 1.
- Recorded decisions for the open questions in
  [architecture.md](architecture.md#11-open-design-questions).

### Exit criteria

- [x] A HIP kernel and a rocBLAS GEMM run successfully on the target device.
- [x] rocWMMA (or the chosen WMMA path) compiles and executes a correctness
      check.
- [x] Allocation classes are benchmarked; the weight, activation, and staging
      policies have measured justification.
- [x] The Go/C ABI round trip works for allocation, submission, event wait, and
      error propagation.
- [x] The build is reproducible from a clean checkout.
- [x] The hardware report records ROCm, driver, firmware, and power profile.

### Risks and dependencies

- gfx1151 support may require a specific ROCm release or kernel.
- WMMA support and code generation may differ from expectations.
- UMA allocation behavior may invalidate a planned zero-copy path.
- Thermal or power limits may dominate short benchmarks.

## 4. Phase 1 — Minimal Inference Loop

### Objective

Prove model semantics end-to-end: a prompt produces correct tokens on a single
request, with a GPU path and a CPU reference path.

### Deliverables

- Pure Go GGUF reader with metadata, tensor directory, shard, alignment, and
  validation support.
- Tokenizer and chat-template implementation with token-exact comparison tests.
- Qwen3.5 hybrid, Qwen3 Dense, and Llama Dense model builders.
- CPU reference implementations for the required operators.
- HIP baseline kernels: FP16 GEMM, RMSNorm, RoPE, basic attention, and sampling.
- `ExecutionPlan` construction and a single-request executor.
- CLI path for loading a model, entering a prompt, and generating tokens.
- Layer-by-layer numerical comparison against the reference.

### Exit criteria

- [x] A fixed prompt produces token IDs identical to the reference, or logits
      within the documented tolerance.
- [x] The model generates continuously and stops on EOS or the configured stop
      condition.
- [x] The same prompt and greedy settings produce the same tokens on repeated
      runs.
- [x] The CPU reference and GPU path agree at tensor and logits levels.
- [x] The tokenizer matches the reference for English, Chinese, special tokens,
      and chat templates.
- [x] The Q4_K file used for the target model loads and validates, even if its
      native execution path is not yet enabled.

### Risks and dependencies

- Tokenizer and chat-template mismatch can invalidate every downstream result.
- RoPE variants, Q/K normalization, GQA, and causal masking are common sources
  of silent divergence.
- Numerical tolerance must be defined per stage, not guessed at the end.
- The first model must be small enough for fast regression cycles.

## 5. Phase 2 — Quantized Decode

### Objective

Make decode fast by computing directly on GGUF-quantized weights, without
expanding the model to a full FP16 weight matrix.

### Deliverables

- `gemv_q4_k`, `gemv_q6_k`, and `gemv_q8_0` kernels with fused unpack, scale
  application, multiply-accumulate, and reduction.
- Correct handling of Q4_K superblocks, scales, and mins.
- Workgroup mapping and vectorization tuned for Wave32 and the target CU count.
- Dispatch thresholds for batch sizes 1, 2, 4, 8, and 16.
- Per-kernel correctness and bandwidth analysis.
- Qwen3.5-4B Q4_K_M end-to-end generation.

### Exit criteria

- [x] Qwen3.5-4B Q4_K_M generates correct tokens on the target device.
- [x] Decode does not expand all quantized weights into FP16.
- [x] The decode bottleneck is attributable to a specific kernel and memory
      behavior, with profiler evidence.
- [x] Each quantized kernel has correctness tests across random and non-aligned
      shapes.
- [x] A decode performance baseline is recorded per quantization type and batch
      size.
- [x] The planner falls back cleanly when a quantization/shape combination is
      unsupported.

### Risks and dependencies

- Q4_K scale/min layout and alignment errors can produce plausible but wrong
  logits.
- Occupancy may collapse under register pressure.
- A kernel that wins in isolation may lose end-to-end because of launch overhead.
- Numerical tolerance for low-bit accumulation must be settled before tuning.

## 6. Phase 3 — High-Performance Prefill

### Objective

Raise prompt-processing throughput and establish predictable long-context
performance.

### Deliverables

- rocBLAS/rocWMMA FP16 baseline and a measured comparison point.
- Quantized WMMA GEMM kernels for Q4_K and Q6_K.
- Flash Attention for prefill with causal masking and GQA.
- Chunked prefill support.
- Fusion of dequantization, normalization, RoPE, SwiGLU, and decode attention
  where measurements justify it.
- Offline dispatch tables by shape, quantization, and batch class.
- Benchmark curves for the required prompt lengths.

### Exit criteria

- [x] Performance curves exist for 128, 512, 2048, 8192, and 32768 prompt tokens.
- [x] Results are compared with llama.cpp HIP and Vulkan on the same GGUF and
      hardware conditions.
- [x] Custom quantized GEMM is selected only where it beats the baseline.
- [x] Short-prompt latency does not regress.
- [x] Chunked prefill completes long prompts without corrupting KV state.
- [x] Kernel resource usage (VGPR, LDS, occupancy) is recorded.

### Risks and dependencies

- WMMA layout conversion and LDS pressure can erase dequantization savings.
- Long-context attention may become memory-bound or numerically unstable.
- Custom GEMM may not win for all shapes; dispatch tables must reflect measured
  reality.
- Benchmark noise and thermal throttling can obscure small improvements.

## 7. Phase 4 — Multi-Request Serving

### Objective

Serve concurrent requests with predictable latency, correct cancellation, and
efficient KV reuse.

### Deliverables

- Paged KV cache with block allocator, per-request block table, reference
  counting, and cleanup.
- Continuous batching and scheduler-level chunked prefill interleaving.
- Prefix cache with a documented key and eviction policy.
- HIP Graph capture and replay for stable decode shapes, with fallback.
- SSE streaming and an OpenAI-compatible chat/completions endpoint.
- Health, metrics, and profiling endpoints.
- Request cancellation, timeout, and backpressure handling.

### Exit criteria

- [x] Single-user latency and multi-request throughput are reproducible.
      `bench/serving` records queue, TTFT, prefill, decode, and sampling
      percentiles plus aggregate throughput with full metadata; the Phase 4
      table is in [benchmarks.md](benchmarks.md).
- [x] Concurrent requests produce outputs identical to isolated runs under the
      same sampling settings. CPU and HIP serving tests run the same prompts at
      concurrency 1 and 2 and compare tokens and text exactly.
- [x] Cancellation and failures return all KV blocks and buffers. Cancellation
      tests assert allocator `Used == 0` and balanced allocation/release counts
      for both backends.
- [x] Repeated request cycles show no unbounded memory growth. Twelve sequential
      cycles leave zero used blocks and no active requests.
- [x] Graph replay falls back correctly when shapes, addresses, or block tables
      cannot be updated. Native tests cover capture/replay with dynamic
      positions and rejection of non-dynamic plans; a Go test invalidates the
      dynamic buffer and verifies normal execution matches a plain session.
- [x] API contract tests cover streaming, errors, and stop conditions. Tests
      cover SSE role/delta/finish/[DONE] events, stop strings, strict parameter
      rejection, structured errors, health, metrics, pprof, and client
      cancellation.
- [x] Metrics distinguish queue wait, prefill, decode, and sampling time.
      Scheduler and per-request results record the four phases separately, and
      `/metrics` exports totals plus TTFT/TPOT percentiles.

### Implementation notes and limits

- Continuous batching is iteration-level: the scheduler recomposes the active
  set between steps and records each batch, but each request executes its own
  plan on the shared stream. Mixed prefill/decode or multi-request decode
  kernels are not implemented, so aggregate throughput stays near the
  single-request rate.
- KV defaults to FP32. Phase 5 adds opt-in F16 and Q8_0 formats through
  `--kv-type`; see the Phase 5 notes below.
- Prefix entries cache KV blocks, the recurrent state snapshot, and prompt
  logits. Sharing a partially filled block copies it before the request writes
  past the cached prefix.
- HIP Graph capture is opt-in. The measured decode difference on Qwen3.5-4B
  Q4_K_M at 256 context was within noise, so the plain stream path is the
  default; capture, parameter update, and replay failures fall back to it.
- The server loads one model. Prefix cache keys include the model identity so a
  future multi-model server cannot reuse entries across models.

### Risks and dependencies

- Scheduler fairness and starvation are easy to get wrong under mixed prefill
  and decode load.
- KV fragmentation and prefix-cache eviction can cause capacity surprises.
- Graph replay parameter updates are a common source of stale-pointer bugs.
- API compatibility needs an explicit supported-subset definition.

## 8. Phase 5 — Expansion and Research

### Objective

Broaden model coverage and evaluate advanced execution strategies without
destabilizing the core GPU path.

### Deliverables

- MoE routing and grouped GEMM for Qwen3 MoE and similar models.
- KV-cache quantization.
- Speculative decoding experiments.
- NPU prefill/decode and subgraph experiments under
  [npu.md](npu.md).
- Additional GGUF architectures and quantization types driven by demand.

### Exit criteria

- [x] Each feature is behind a capability check or explicit flag.
      KV format is selected by `--kv-type`, speculative decoding by
      `--speculative`, NPU by `--npu`, and MoE/quantization support by GGUF
      architecture and dtype capability checks.
- [x] Each executable feature has correctness evidence against the CPU
      reference and a measured performance result. See the Phase 5 section of
      [benchmarks.md](benchmarks.md). NPU execution is not claimed: the probe
      result and fallback are the Phase 5 NPU evidence.
- [x] No feature regresses the accepted GPU-only path. Phase 5 features are
      additive or opt-in; F32 KV and plain decode remain the defaults.
- [x] NPU paths are opt-in and fall back to the selected backend. The target host reports
      `device_only`; the probe result and fallback are recorded below.
- [x] Documentation and benchmark metadata are updated.

### Implementation notes and limits

- `qwen3moe` is a CPU reference and HIP decode/prefill architecture. Routing
  selects the top-k experts deterministically, grouped HIP kernels bin
  assignments by expert, and the expert weights stay packed. The HIP path is
  validated on synthetic GGUF fixtures because no standard `qwen3moe` model is
  available on the target host.
- KV formats are F32 (default), F16, and Q8_0. Q8_0 stores one FP16 scale per
  32-value group. Prefix-cache keys include the format. KV format does not
  change decode latency measurably at 128 prompt / 32 decode tokens; its benefit
  is cache memory.
- Speculative decoding is exact greedy prompt-lookup speculation. It is not
  available for non-greedy sampling or the MoE HIP runtime, and it is exposed
  only through the CLI/bench path.
- Q4_0 and Q5_0 are supported by the GGUF reader, CPU reference, HIP GEMV,
  embedding, and WMMA GEMM. Q4_1/Q5_1/Q2_K/Q3_K/quantized K-cache remain
  unsupported.
- NPU: the probe found `/dev/accel/accel0` and the `amdxdna` module but no XRT
  userspace runtime. No NPU kernel, graph compiler, or subgraph partition is
  shipped; `--npu` records this result and falls back to the selected GPU
  backend. Activation criteria in [npu.md](npu.md) are not met.

### Risks and dependencies

- Scope can expand faster than correctness coverage.
- MoE and MLA introduce new memory layouts and routing kernels.
- Speculative decoding changes sampling semantics and needs careful validation.
- NPU toolchain maturity and memory cost may make acceleration impractical.

## 9. Milestones and definition of done

| Milestone | Demonstrable artifact | Evidence |
| --- | --- | --- |
| **M0: Toolchain proven** | HIP, rocBLAS, and WMMA smoke tests; allocation report | Phase 0 checklist |
| **M1: Correct generation** | Single-request Qwen3/Llama generation with CPU/GPU agreement | Token and logits comparison report |
| **M2: Quantized decode** | Qwen3.5-4B Q4_K_M generation with native GEMV | Per-kernel correctness and decode baseline |
| **M3: Prefill performance** | Prompt-length performance curves | Benchmark report vs llama.cpp |
| **M4: Concurrent service** | Paged KV, continuous batching, SSE API | Latency/throughput report and leak checks |
| **M5: Expansion** | At least one new architecture/feature with measured benefit | Feature report and regression run |

A milestone is "done" only when its evidence is checked in or linked from the
repository and can be reproduced by another contributor with the documented
environment.

## 10. Cross-cutting tracks

| Track | Ongoing work |
| --- | --- |
| **Correctness** | Golden token tests, logits tolerances, tensor-level references, edge-shape tests, cross-framework comparison |
| **Performance** | Per-kernel benchmarks, dispatch tables, profiler reports, regression tracking, thermal metadata |
| **Observability** | Phase timings, queue and KV metrics, kernel attribution, error classification |
| **Documentation** | Keep architecture, format, kernel, serving, and test docs synchronized with implementation |
| **Build and release** | Reproducible Go/CMake builds, C ABI versioning, CI tiers, fixture management |

These tracks are not separate phases. Each phase must leave them in a consistent
state.

## 11. Dependencies and critical path

```text
Hardware/toolchain validation
        |
GGUF reader + tokenizer + model config
        |
CPU reference + numerical tolerances
        |
HIP baseline execution
        |
Quantized decode
        |
Quantized prefill + attention
        |
Paged KV + continuous batching
        |
Graph replay + API hardening
        |
Architecture and NPU expansion
```

Critical risks early in the path:

1. **Tokenizer correctness** — a mismatch invalidates token-level acceptance.
2. **Numerical baseline** — without stage tolerances, kernel work cannot be
   judged.
3. **Hardware characterization** — allocation and bandwidth assumptions shape
   the memory planner.
4. **Quantized GEMV correctness** — the primary decode optimization depends on
   exact Q4_K layout handling.

A phase may begin exploratory work before the previous gate closes, but it must
not declare completion or depend on unverified results from an open gate.

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| ROCm/gfx1151 support gaps | Toolchain or kernels unavailable | Phase 0 validation; pin a known-good ROCm version; document fallbacks |
| UMA allocation assumptions | Wrong memory policy and poor bandwidth | Measure allocation classes; keep policy configurable |
| WMMA/rocWMMA limitations | Prefill path blocked or slow | Maintain rocBLAS baseline; design kernels around available instructions |
| Q4_K layout bugs | Plausible but incorrect output | Byte-level decode tests; CPU reference; cross-framework comparison |
| Tokenizer/chat-template mismatch | Invalid end-to-end results | Token-exact golden tests before model work |
| Numerical divergence | Silent quality loss | Stage-wise tolerances; deterministic test mode; independent reference |
| Scheduler complexity | Latency spikes, starvation, leaks | Start single-request; add paged KV and batching incrementally; leak tests |
| Benchmark noise/thermals | Misleading optimization decisions | Fixed protocol, repetitions, percentiles, power/clock metadata |
| NPU toolchain immaturity | Wasted effort | Defer; gate on measured end-to-end benefit and opt-in fallback |
| Scope creep | Core path unfinished | Enforce phase gates and non-goals |

## 13. Out of scope for the first release

- Training or automatic differentiation.
- Coverage of every GGUF architecture and quantization type.
- Distributed inference or multi-GPU tensor parallelism.
- Automatic fine-grained multi-device operator partitioning.
- A general-purpose graph compiler.
- Multi-model serving and model swapping.
- NPU-dependent execution paths.

## 14. Success criteria for the first release

The first release is successful when all of the following hold. Checked items
are already satisfied by the current single-request vertical slice; unchecked
items remain release gates.

- [x] Qwen3.5-4B Q4_K_M GGUF loads and validates on the target gfx1151 system.
- [x] A single request generates tokens end-to-end on the GPU.
- [x] Tokenizer, chat template, and greedy token sequences match the reference
      implementation.
- [x] Quantized decode runs natively, without expanding the model to FP16.
- [ ] Prefill and decode performance are measured across the required prompt and
      output lengths.
- [x] The CPU reference path remains available for correctness and fallback.
- [x] Results are reproducible from the documented environment and benchmark
      metadata.
- [x] The architecture, format, kernel, serving, and testing documents match the
      implemented single-request behavior; later-phase behavior remains
      documented as target design.

Phase 5 exit criteria are met; the first release remains gated by the
decode-length performance matrix. Prefill lengths are covered by the Phase 3
curves, and Phase 4 serving evidence is recorded in this document and
[benchmarks.md](benchmarks.md). NPU acceleration is not claimed because the
target host lacks an XRT userspace runtime; the opt-in probe and GPU fallback are
documented in [npu.md](npu.md).
