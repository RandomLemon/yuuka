# yuuka Documentation

`yuuka` is a from-scratch LLM inference framework for AMD **gfx1151** (RDNA 3.5)
APUs. The runtime is written in **Go**; performance-critical device code is
written in **HIP/C++**. It loads GGUF models directly and does not use
llama.cpp, GGML, or Ollama as an inference execution kernel. Those projects are
treated as format and correctness references only.

## Scope and status

The runtime provides CPU reference execution and HIP execution for the
`qwen35` dense hybrid and `qwen35moe` hybrid MoE architectures, chunked
prefill, paged KV with iteration-level continuous scheduling, prefix caching,
optional HIP Graph replay, a CLI, and an OpenAI-compatible server. The CPU
reference path is the default; the HIP path requires the native shared library
and a `-tags hip` Go build.

Model architectures, quantization types, and KV formats outside the sets
defined in [gguf.md](gguf.md) and [kernels.md](kernels.md) are rejected
explicitly. NPU support is a future task; the existing `npu-probe`/`--npu` path
is diagnostic only and does not execute model operations ([npu.md](npu.md)).

Measured results, the evidence for each measurement session, and the current
limits are recorded only in the model reports indexed by
[benchmarks.md](benchmarks.md). Measured hardware and toolchain facts are in
[hardware.md](hardware.md). Verification is layered; tests that need a real
model or a GPU are opt-in and skip explicitly rather than passing silently
([testing.md](testing.md)).

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

1. Load mainstream GGUF models, starting with the Qwen3.5 hybrid decoder.
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
| [architecture.md](architecture.md) | Layered design, core interfaces, Go/HIP boundary, memory model, key decisions | All contributors |
| [gguf.md](gguf.md) | GGUF reader contract, metadata mapping, tokenizer, quantization and architecture scope | Runtime and model contributors |
| [kernels.md](kernels.md) | Operator families, supported dtypes, selection rules, fusion, numerics | Kernel contributors |
| [serving.md](serving.md) | Scheduler, continuous batching, paged KV cache, HIP Graph, API and observability | Runtime and service contributors |
| [testing.md](testing.md) | Correctness strategy, test matrix, benchmark protocol, regression gates | All contributors |
| [npu.md](npu.md) | Deferred XDNA 2 NPU strategy, interface reservations, open questions | Architecture and research contributors |
| [hardware.md](hardware.md) | Measured gfx1151 toolchain, memory, rocBLAS, and rocWMMA facts | All contributors |
| [benchmarks.md](benchmarks.md) | Benchmark report conventions, required metadata and coverage, and the per-model report index | All contributors |

### Module indexes

- [Qwen3.5-4B benchmark](benchmarks/qwen3.5_4b.md) — The single-request `qwen35` HIP record: prefill and decode curves, KV formats, isolated kernel sweeps, serving, the llama.cpp comparison, and the measured dispatch decisions.
- [Qwen3.6-35B-A3B benchmark](benchmarks/qwen3.6_35b_a3b.md) — The matched 128/512/2048/8192-token `qwen35moe` HIP prefill comparison next to llama.cpp, the per-operation attribution, the assignment-tile table, and the correctness gates.

## Reading paths

- **New contributor:** [README](README.md) → [architecture](architecture.md) →
  the document for your area → [testing](testing.md).
- **Hardware bring-up:** [hardware](hardware.md) → [benchmarks](benchmarks.md) →
  [testing](testing.md).
- **Kernel work:** [kernels](kernels.md) → [testing](testing.md) →
  [benchmarks](benchmarks.md).
- **Runtime/service work:** [architecture](architecture.md) →
  [serving](serving.md) → [benchmarks](benchmarks.md).
- **Model/format work:** [gguf](gguf.md) → [testing](testing.md) →
  [architecture](architecture.md).

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
  targets. Only results recorded under the protocol in [testing.md](testing.md)
  and published in a model report listed in [benchmarks.md](benchmarks.md) count
  as benchmarks.
- Design documents state contracts and link to the owning report instead of
  restating measured values, kernel names, or file layout.
- Diagrams are intentionally textual so they remain readable in terminals,
  code review, and generated documentation.
- All measurements must record the model file, quantization, ROCm/driver
  version, sampling parameters, warmup, and thermal/power conditions.
