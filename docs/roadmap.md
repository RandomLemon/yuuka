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
tested; CPU and HIP greedy tokens match, and the recorded first-step maximum
absolute logit difference is `4.58e-5`.

Current gaps: Phase 0 has checked-in CI tiers and a generated tiny GGUF fixture,
but the firmware/power/thermal record is incomplete. Phase 1 covers only
`qwen35`; Qwen3 Dense, Llama Dense, and FP16 GEMM are not implemented. Phase 2
has quantized GEMV and end-to-end Q4_K_M generation, but no profiler
attribution, batch dispatch thresholds, or per-shape sweep. WMMA/rocBLAS
prefill, paged KV, batching, and prefix caching are not implemented yet. The
current measured baseline is in [benchmarks.md](benchmarks.md).

This status does not declare any phase exit gate complete. It identifies the
current boundary for contributors.

## 2. Phase overview

| Phase | Objective | Key deliverables | Exit gate |
| --- | --- | --- | --- |
| **0. Bootstrap** | Prove the toolchain and characterize the device | Hardware report, repository scaffold, C ABI, GGUF fixtures, CPU reference skeleton | HIP/rocBLAS/WMMA smoke tests and allocation/bandwidth measurements recorded |
| **1. Minimal loop** | Correct single-request generation | GGUF reader, tokenizer, Qwen3/Llama dense, CPU reference ops, HIP FP16 baseline, CLI | Fixed-prompt token IDs match or logits are within documented tolerance; generation stops correctly |
| **2. Quantized decode** | Native low-bit decode | Q4_K/Q6_K/Q8_0 GEMV, fused unpack/scale, dispatch thresholds, profiling | Qwen3 Q4_K runs correctly; decode bottleneck is attributable to a kernel and memory behavior |
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

- [ ] A HIP kernel and a rocBLAS GEMM run successfully on the target device.
- [ ] rocWMMA (or the chosen WMMA path) compiles and executes a correctness
      check.
- [ ] Allocation classes are benchmarked; the weight, activation, and staging
      policies have measured justification.
- [ ] The Go/C ABI round trip works for allocation, submission, event wait, and
      error propagation.
- [ ] The build is reproducible from a clean checkout.
- [ ] The hardware report records ROCm, driver, firmware, and power profile.

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

- [ ] A fixed prompt produces token IDs identical to the reference, or logits
      within the documented tolerance.
- [ ] The model generates continuously and stops on EOS or the configured stop
      condition.
- [ ] The same prompt and greedy settings produce the same tokens on repeated
      runs.
- [ ] The CPU reference and GPU path agree at tensor and logits levels.
- [ ] The tokenizer matches the reference for English, Chinese, special tokens,
      and chat templates.
- [ ] The Q4_K file used for the target model loads and validates, even if its
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

- [ ] Qwen3 Q4_K generates correct tokens on the target device.
- [ ] Decode does not expand all quantized weights into FP16.
- [ ] The decode bottleneck is attributable to a specific kernel and memory
      behavior, with profiler evidence.
- [ ] Each quantized kernel has correctness tests across random and non-aligned
      shapes.
- [ ] A decode performance baseline is recorded per quantization type and batch
      size.
- [ ] The planner falls back cleanly when a quantization/shape combination is
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

- [ ] Performance curves exist for 128, 512, 2048, 8192, and 32768 prompt tokens.
- [ ] Results are compared with llama.cpp HIP and Vulkan on the same GGUF and
      hardware conditions.
- [ ] Custom quantized GEMM is selected only where it beats the baseline.
- [ ] Short-prompt latency does not regress.
- [ ] Chunked prefill completes long prompts without corrupting KV state.
- [ ] Kernel resource usage (VGPR, LDS, occupancy) is recorded.

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
- Continuous batching and chunked prefill scheduling.
- Prefix cache with a documented key and eviction policy.
- HIP Graph capture and replay for stable decode shapes, with fallback.
- SSE streaming and an OpenAI-compatible chat/completions endpoint.
- Health, metrics, and profiling endpoints.
- Request cancellation, timeout, and backpressure handling.

### Exit criteria

- [ ] Single-user latency and multi-request throughput are reproducible.
- [ ] Concurrent requests produce outputs identical to isolated runs under the
      same sampling settings.
- [ ] Cancellation and failures return all KV blocks and buffers.
- [ ] Repeated request cycles show no unbounded memory growth.
- [ ] Graph replay falls back correctly when shapes, addresses, or block tables
      cannot be updated.
- [ ] API contract tests cover streaming, errors, and stop conditions.
- [ ] Metrics distinguish queue wait, prefill, decode, and sampling time.

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

- [ ] Each feature is behind a capability check or explicit flag.
- [ ] Each feature has correctness evidence against the CPU reference and a
      measured performance result.
- [ ] No feature regresses the accepted GPU-only path.
- [ ] NPU paths are opt-in and fall back to GPU.
- [ ] Documentation and benchmark metadata are updated.

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
| **M2: Quantized decode** | Qwen3 Q4_K generation with native GEMV | Per-kernel correctness and decode baseline |
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

The first release is successful when:

- [ ] Qwen3.5-4B Q4_K_M GGUF loads and validates on the target gfx1151 system.
- [ ] A single request generates tokens end-to-end on the GPU.
- [ ] Tokenizer, chat template, and greedy token sequences match the reference
      implementation.
- [ ] Quantized decode runs natively, without expanding the model to FP16.
- [ ] Prefill and decode performance are measured across the required prompt and
      output lengths.
- [ ] The CPU reference path remains available for correctness and fallback.
- [ ] Results are reproducible from the documented environment and benchmark
      metadata.
- [ ] The architecture, format, kernel, serving, and testing documents match the
      implemented behavior.
