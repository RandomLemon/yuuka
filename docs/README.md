# yuuka Documentation

`yuuka` is a from-scratch LLM inference framework for AMD **gfx1151** (RDNA 3.5)
APUs. The runtime is written in **Go**; performance-critical device code is
written in **HIP/C++**. It loads GGUF models directly and does not use
llama.cpp, GGML, or Ollama as an inference execution kernel. Those projects are
treated as format and correctness references only.

> **Status: CPU and HIP single-request vertical slice with chunked prefill and
> concurrent paged serving implemented.** The repository contains a validated
> GGUF reader, a Qwen3.5 tokenizer, a CPU reference forward pass, a HIP device
> executor for the same decoder, rocWMMA/rocBLAS prefill GEMM, paged Flash
> Attention, chunked prefill, a continuous scheduler, a reference-counted
> paged KV and prefix cache, opt-in HIP Graph replay with fallback, a CLI, and
> an OpenAI-compatible SSE server with health, metrics, and pprof endpoints.
> Phase 5 adds `qwen3moe` grouped expert execution, F16/Q8_0 KV formats, greedy
> prompt-lookup speculative decoding, Q4_0/Q5_0 tensors, and an opt-in NPU
> capability probe with explicit fallback. NPU execution itself is not enabled
> because the target host has no XRT userspace runtime. See the implementation
> status below, [hardware.md](hardware.md), and [benchmarks.md](benchmarks.md).

## Target configuration

| Item | Target |
| --- | --- |
| Hardware | AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151, RDNA 3.5) |
| Compute | 40 CUs in the reference configuration; Wave32-first |
| Memory | LPDDR5x system memory shared between CPU and GPU (UMA APU) |
| OS / stack | Linux with ROCm/HIP; rocBLAS and rocWMMA allowed |
| Model format | GGUF |
| First model | Qwen3.5-4B hybrid (`qwen35`), Q4_K_M GGUF |
| First milestone | Single-request, GPU-only inference from prompt to generated tokens |

Hardware capabilities must be verified on the actual device, kernel, and
driver stack. GPU-visible memory, mappable host memory, rocWMMA availability,
and NPU Linux runtime support are not assumed from marketing specifications.

## Goals

1. Load mainstream GGUF models, starting with dense decoder Transformers.
2. Execute the full Transformer on the GPU: quantized weight matrix
   multiplication, attention, normalization, activations, and sampling.
3. Design separate prefill and decode kernel families, dispatched by shape,
   quantization type, and batch size.
4. Start with low-latency single-request inference, then add continuous
   batching and paged KV cache.
5. Keep Go responsible for model parsing, execution planning, request
   scheduling, memory lifetime, APIs, and observability.
6. Retain a CPU reference backend for correctness. Add an NPU backend later
   without making the GPU-only path depend on it.

## Non-goals for the first release

- Training or automatic differentiation.
- Covering every GGUF model architecture and quantization type.
- Automatic fine-grained operator partitioning across multiple devices.
- Distributed inference or multi-GPU tensor parallelism.
- A heavy general-purpose graph compiler.

## Design principles

- **Correctness before speed.** Establish a reproducible numerical baseline
  before optimizing kernels.
- **Keep quantized weights quantized.** Do not expand an entire quantized model
  into FP16 weights for decode.
- **Prefill and decode are different problems.** They use different operator
  families, tiling, and dispatch policies, not merely different parameters to
  the same GEMM.
- **Measure, do not assume.** Validate bandwidth, allocation behavior,
  occupancy, and kernel wins on gfx1151 with recorded configurations.
- **Stable seams.** GGUF parsing, model semantics, operator interfaces, device
  implementations, and request scheduling are independent layers.
- **Boring runtime.** Prefer an explicit, inspectable `ExecutionPlan` and
  replaceable backends over a general graph compiler.

## Document map

| Document | Contents | Primary audience |
| --- | --- | --- |
| [architecture.md](architecture.md) | Layered design, core interfaces, Go/HIP boundary, memory model, repository layout, key decisions | All contributors |
| [gguf.md](gguf.md) | GGUF reader, metadata mapping, tokenizer, quantization handling, model support matrix | Runtime and model contributors |
| [kernels.md](kernels.md) | Prefill/decode kernel design, quantized GEMV/GEMM, attention, fusion, dispatch and tuning | Kernel contributors |
| [serving.md](serving.md) | Scheduler, continuous batching, paged KV cache, HIP Graph, API and observability | Runtime and service contributors |
| [testing.md](testing.md) | Correctness strategy, test matrix, benchmark protocol, reporting, regression gates | All contributors |
| [npu.md](npu.md) | Deferred XDNA 2 NPU strategy, interface reservations, open questions | Architecture and research contributors |
| [roadmap.md](roadmap.md) | Phase-by-phase plan, deliverables, exit criteria, risks, dependencies | Maintainers and reviewers |
| [hardware.md](hardware.md) | Measured gfx1151 toolchain, memory, rocBLAS, and rocWMMA facts | All contributors |
| [benchmarks.md](benchmarks.md) | Single-request, prefill, and concurrent-serving measurements with reproduction steps | All contributors |

## Implementation status

Implemented and tested:

- Pure Go GGUF reader with header, metadata, tensor directory, split shards,
  alignment, overlap, and byte-range validation.
- Qwen3.5 byte-level BPE and Llama SentencePiece tokenizers, with golden token
  ID tests against llama.cpp for English, Chinese, whitespace, punctuation,
  emoji, special tokens, and chat prompts.
- CPU reference decoder for the `qwen35` hybrid architecture: full GQA
  attention, gated delta-net linear attention, convolution state, RMSNorm,
  RoPE, SwiGLU, tied LM head, and KV/recurrent caches.
- CPU reference decoders for dense Llama, dense Qwen3, and sparse `qwen3moe`
  layouts, including optional Q/K norm, tied or untied output heads, top-k
  expert routing, and generated tiny fixtures.
- Deterministic greedy, top-k, and nucleus sampling.
- `yuuka inspect`, `yuuka run`, and `yuuka bench`.
- `yuuka-server` with `/v1/chat/completions`, SSE streaming, stop sequences,
  health, metrics, cancellation, and explicit busy rejection.
- HIP C ABI for devices, buffers, streams, events, transfers, planned
  operations, quantized GEMV, a batched F32/F16 GEMM baseline, and the chunked
  prefill operator family; validated `F16`, `Q8_0`, `Q4_0`, `Q5_0`, `Q4_K`,
  `Q5_K`, and `Q6_K` GEMV kernels, embeddings, and WMMA GEMM.
- rocWMMA FP16 and quantized prefill GEMM, a rocBLAS FP16 baseline, causal GQA
  Flash Attention, chunked prefill with per-chunk KV writes, and the measured
  `SelectPrefillPath` dispatch table.
- HIP execution of the full Qwen3.5 decoder: embedding, normalization, RoPE,
  full attention, gated delta-net, convolution state, SwiGLU, and the LM head.
  The CLI and server select it with `--backend hip` when built with
  `-tags hip`.
- Batched quantized GEMV with measured dispatch thresholds, a per-operation HIP
  event profile path, and per-quantization/per-batch decode baselines.
- End-to-end GPU/CPU comparison on the validated model: greedy token sequences
  match. The decode path's recorded first-step maximum absolute logit
  difference is `4.58e-5`; the chunked prefill path reports `3.65e-3` on the
  same prompt with the same top-1 token. The checked-in tests enforce the token
  sequence, a loose logit bound, and a layer-by-layer hidden-state comparison;
  the worst recorded first-prompt layer difference is `5.53e-5`. Chunked
  prefill at 1024 tokens produces bitwise identical logits across chunk sizes
  128, 512, and 1024.
- Phase 0 hardware probe covering allocation, copy, memset, mapped memory,
  rocBLAS SGEMM, and a rocWMMA FP16 tile.
- Prefill performance curves at 128-32768 prompt tokens, resource usage, and a
  same-GGUF comparison with llama.cpp HIP and Vulkan.
- Paged KV with a block allocator, continuous batching, a collision-safe prefix
  cache, opt-in HIP Graph replay, and an OpenAI-compatible server.
- Opt-in F16 and Q8_0 KV formats with CPU and HIP write/attention kernels.
- `qwen3moe` HIP execution with top-k routing, per-expert assignment buckets,
  and grouped expert kernels that keep expert weights packed.
- Exact greedy prompt-lookup speculative decoding, verified against the plain
  greedy path and enabled with `--speculative`.
- An opt-in NPU capability probe (`yuuka npu-probe`, `--npu`) that reports
  `device_only` on the reference host and falls back to the selected backend.

Not yet implemented:

- HIP execution for dense Llama/Qwen3 (non-MoE) models.
- NPU execution kernels, graph compilation, and subgraph partitioning; the
  reference host lacks an XRT userspace runtime.
- Quantization presets and architectures outside the tested set, including
  Q4_1/Q5_1, Q2_K/Q3_K, Q8_K, and the IQ families.
- Speculative decoding for non-greedy sampling or the MoE HIP runtime.

Verification gaps: real-model tests are opt-in through `YUUKA_TEST_MODEL`; GPU
tests require the reference device. Unit tests that do not need the real model
use the deterministic tiny GGUF generated by `tests/fixture`, so no model
weights are committed.

The CPU reference path is the default. The HIP path requires the native shared
library and a `-tags hip` Go build. `general.architecture = qwen35` is the primary model family; `qwen3moe` has
HIP support, and dense `llama`/`qwen3` remain CPU reference paths.

## Reading paths

- **New contributor:** [README](README.md) → [architecture](architecture.md) →
  the document for your area → [roadmap](roadmap.md).
- **Hardware bring-up:** [hardware](hardware.md) → [roadmap Phase 0](roadmap.md#3-phase-0--bootstrap-and-hardware-validation).
- **Kernel work:** [kernels](kernels.md) → [testing](testing.md) →
  [roadmap Phase 2–3](roadmap.md#5-phase-2--quantized-decode).
- **Runtime/service work:** [architecture](architecture.md) →
  [serving](serving.md) → [roadmap Phase 4](roadmap.md#7-phase-4--multi-request-serving).
- **Model/format work:** [gguf](gguf.md) → [testing](testing.md) →
  [roadmap Phase 1](roadmap.md#4-phase-1--minimal-inference-loop).

## Glossary

| Term | Meaning |
| --- | --- |
| **Prefill** | Processing the prompt in one or more large matrix multiplications; typically compute-throughput limited. |
| **Decode** | Generating one token (or a small batch) at a time; typically limited by weight-read bandwidth and kernel launch overhead. |
| **GEMV / GEMM** | Matrix-vector and matrix-matrix multiplication. Decode maps to GEMV; prefill maps to GEMM. |
| **WMMA** | RDNA matrix instructions used for the prefill GEMM path. CDNA MFMA kernels are not portable to this target. |
| **GQA** | Grouped-query attention; multiple query heads share a KV head. |
| **KV cache** | Cached key/value tensors for previously processed tokens. |
| **Paged KV** | KV cache managed as fixed-size blocks referenced through a per-request block table. |
| **TTFT** | Time to first token. |
| **TPOT** | Time per output token during decode. |
| **UMA** | Unified memory architecture; CPU and GPU share physical memory, but allocation paths can still differ in performance. |
| **ExecutionPlan** | The Go-side, model-shaped plan that binds weights, kernels, scratch buffers, and streams before execution. |

## Documentation conventions

- `MUST`, `SHOULD`, and `MAY` describe design requirements, not current
  implementation behavior.
- Performance numbers in these documents are design estimates or measurement
  targets. Only results recorded under the protocol in
  [testing.md](testing.md) count as benchmarks.
- Diagrams are intentionally textual so they remain readable in terminals,
  code review, and generated documentation.
- All measurements must record the model file, quantization, ROCm/driver
  version, sampling parameters, warmup, and thermal/power conditions.
