# Architecture

## 1. Purpose and scope

This document describes the intended structure of `yuuka`: a GGUF inference
framework with a Go runtime and HIP device kernels, targeting a single AMD
gfx1151 APU. It covers the layer boundaries, the core interfaces, the Go/HIP
seam, memory ownership, backend selection, and the rules that keep the layers
independent.

It specifies design intent and the contracts each layer must satisfy, not
current implementation state. Release scope and status are summarized in
[README.md](README.md); kernel algorithms are specified in
[kernels.md](kernels.md); request scheduling and serving are specified in
[serving.md](serving.md); format handling is specified in
[gguf.md](gguf.md).

## 2. Target hardware and constraints

The primary target is **gfx1151** (RDNA 3.5), using Ryzen AI Max+ 395 /
Radeon 8060S as the reference configuration.

| Hardware feature | Architectural consequence |
| --- | --- |
| 40 CUs in the reference configuration | Keep enough tiles and waves in flight to occupy the GPU; avoid designs that only fill a fraction of CUs. |
| Wave32-friendly execution | Optimize for Wave32 first. |
| RDNA WMMA | Use matrix instructions for prefill GEMM. Do not port CDNA MFMA kernels unchanged. |
| LDS / VGPR / occupancy limits | Control tile size and fusion depth; avoid register spills and occupancy collapse. |
| Shared CPU/GPU physical memory (UMA APU) | Measure HIP and host allocation strategies per workload; do not assume zero-copy is always fastest. |
| LPDDR5x system-memory bandwidth | Single-request decode is often limited by weight-read bandwidth, not arithmetic throughput. |
| XDNA 2 NPU | Treat as an optional separate backend; the GPU path must not depend on it. See [npu.md](npu.md). |

Hardware specifications and ROCm support must be verified on the actual device,
kernel, and driver versions. GPU-visible memory, mappable host memory,
rocWMMA availability, and NPU Linux runtime support are measured facts, not
assumptions.

### 2.1 Prefill and decode asymmetry

Prefill computes a matrix multiplication with a relatively large `M`:

```text
Y[M, N] = X[M, K] * W[K, N]
```

Throughput depends on effective matrix throughput, weight reuse, attention I/O,
and tile scheduling. Decode uses `M = 1` or a small batch, so the critical path
becomes quantized GEMV, weight-read bandwidth, KV-cache reads, and kernel launch
overhead. The two paths therefore have separate operator families, tiling
strategies, and dispatch rules. See [kernels.md](kernels.md).

## 3. Layered architecture

```text
HTTP / CLI / Streaming / OpenAI-compatible API
                   |
Tokenizer / Chat Template / Sampling / Model Manager (Go)
                   |
Request Scheduler / Admission / Iteration-level Continuous Scheduling (Go)
                   |
Model Executor / Prefill / Decode / Paged KV (Go)
                   |
Execution Planner / Operator Dispatch / Allocation (Go)
                   |
             Backend Interface
           /         |          \
      HIP GPU     CPU Ref.     XDNA NPU (future)
           |
   HIP Streams / Events / Graph / Allocators
           |
   Quant GEMV | Quant GEMM | Flash/Paged Attention
   Norm | RoPE | Activation | Fusion
```

### 3.1 Layer responsibilities

| Layer | Owns | Must not own |
| --- | --- | --- |
| API and serving | HTTP/CLI, request parsing, SSE streaming, health and metrics | Model semantics, device memory layout |
| Model and tokenizer | GGUF metadata interpretation, tokenizer, chat template, sampling policy | HIP calls, request scheduling |
| Scheduler | Admission, batching, token/KV budgets, cancellation | Kernel selection, weight layout |
| Executor | Prefill/decode orchestration, KV block tables, sampling loop | Format parsing, transport concerns |
| Planner | Planned operator sequence, operator selection, scratch sizing, kernel signatures | Device-specific kernel internals |
| Backend | Device allocation, streams, events, kernel launch, graph capture | GGUF parsing, model architecture decisions |
| Kernels | Device-side computation | Model metadata, request lifecycle |

**Dependency rule:** higher layers depend on interfaces, not on HIP. A
replacement kernel or backend must not require changes to the GGUF reader or
model definitions.

## 4. Core interfaces

The following types define the intended seams. They are a design shape, not a
frozen API: names, packaging, and concrete implementations may change as long as
the contracts in section 4.1 hold.

```go
type Model interface {
    Config() ModelConfig
    Tokenizer() Tokenizer
    Weights() WeightStore
    BuildPlan(Backend) (Executable, error)
}

type TensorDesc struct {
    Name   string
    Shape  []int64
    DType  DType
    Offset uint64
    Bytes  uint64
}

type Backend interface {
    Name() string
    Capabilities() Capabilities
    Alloc(size uint64) (Buffer, error)
    Compile(plan *ExecutionPlan) (Executable, error)
    Synchronize() error
    Close() error
}

type Executable interface {
    Run(batch *Batch) (Event, error)
    Close() error
}

type Event interface {
    Wait() error
}
```

`TensorDesc` is the only tensor metadata shape the planner and kernels may
consume: a name, logical shape, dtype, and byte range inside the weight store.
The model layer must not expose raw packed bytes upward, and device code must not
inspect model metadata.

### 4.1 Interface contracts

These contracts must be explicit in every implementation:

- **Buffer ownership.** Every `Buffer` has one owner. Transfer of ownership and
  the point of release are part of the API, not an implementation detail.
- **Event lifetime.** An event represents completion of one submission. A
  buffer or scratch region may be reused only after the corresponding event has
  completed. `Wait` is idempotent after completion and may be called more than
  once.
- **Error propagation.** Device errors are returned as Go errors with enough
  context to identify the stage, kernel, and shape. Errors must not cross the C
  ABI as panics or C++ exceptions.
- **Device selection.** The target device is chosen when the backend is
  constructed, not implicitly per operator.
- **Replay validity.** Graph replay must validate buffer addresses, shapes, and
  block-table parameters before reuse, or fall back to normal submission.
- **Idempotent cleanup.** `Close` on an executable must release its resources
  exactly once, including partially initialized plans.

### 4.2 Planning model

Planning inspects the model configuration, backend capabilities, quantization
types, and expected batch shapes before any device allocation. It produces a
plan containing:

- weight buffer bindings and device residency;
- kernel choices and their shape signatures;
- scratch and workspace sizes;
- stream and event structure;
- graph-capture candidates, if supported;
- fallback choices for unsupported operations.

Planning happens at load time and whenever a new shape class requires a new
kernel configuration. It must not perform a cgo call per tensor per token. Chunk
geometry is part of the plan identity: a session caches one prefill plan per
distinct chunk length, the final partial chunk has its own plan, and a change in
chunk geometry invalidates the cached plans.

## 5. Execution model

```text
Open and validate GGUF
        |
Build tokenizer, model config, weight descriptors
        |
Build plan (kernel selection, scratch, residency)
        |
Allocate weights, activation scratch, KV pool, recurrent state
        |
Prefill: prompt chunks -> attention + GEMM -> KV writes
        |
Decode loop: quantized GEMV + paged attention -> logits -> host sampling
        |
Stream tokens / finish / release request resources
```

Key rules:

1. GGUF data is validated before any device allocation.
2. Weights are uploaded once and stay resident for the model lifetime.
3. Prefill and decode share the KV cache but use different operator families:
   prefill runs chunked batched plans, decode runs single-token quantized GEMV
   with per-head attention. Sampling runs on the host over the logits each step
   returns.
4. Sampling policy is a property of the request, not of the device path. A
   future device sampling kernel must fit behind the same interface.
5. Request cancellation releases KV blocks and scratch through the scheduler and
   memory manager, not through ad hoc frees in the API layer. Each request table
   is released exactly once, whether the request completed, was cancelled, or
   failed.
6. The CPU reference path is deliberately simple: it walks the prompt one token
   at a time through the same operators, KV cache, and recurrent-state
   interfaces, so it stays an independent correctness oracle rather than a
   second optimized implementation.

## 6. Go and HIP boundary

Go must not make a fine-grained cgo call for every tensor operation. The
boundary is designed around planned, batched submission:

```text
Go builds/caches the plan
        |
Go submits a batch or decode step
        |
C ABI shim -> HIP stream/graph -> GPU kernels
        |
Event -> Go completion -> sampling/streaming
```

### 6.1 ABI rules

- Compile HIP/C++ into a standalone shared library with a stable C ABI.
- Expose opaque handles and explicit status codes. Do not expose C++ objects,
  templates, or STL types to Go.
- Version the ABI and reject incompatible libraries at startup.
- Keep kernel launch parameters and device pointers inside the backend; Go sees
  handles, sizes, and status values.

### 6.2 Memory safety rules

- GPU asynchronous work must not hold raw pointers into Go-managed memory unless
  the memory is pinned and its lifetime is guaranteed for the entire operation.
- Prefer persistent data in backend-managed buffers, with events coordinating
  reuse and release.
- Host/device transfers must have a defined completion point. A host buffer may
  only be reused once the transfer that reads it has completed.
- Every path that can fail after partial allocation must have a cleanup path
  that releases all acquired buffers and streams.

### 6.3 Target abstractions

| Abstraction | Purpose | Design constraint |
| --- | --- | --- |
| `Device`, `Stream`, `Event` | Execution and synchronization | Explicit streams only; no reliance on a default stream; one event per submission |
| `DeviceBuffer` | Device allocation with explicit lifetime | Single owner, explicit release, no reuse before completion |
| `HostPinnedBuffer`, `MappedBuffer` | Allocation classes with different performance and lifetime rules | Used only where a measurement justifies the class, with lifetime guaranteed for the operation |
| `GraphExecutable`, `GraphReplay` | Cached launch graphs for stable shapes | Validate addresses, shapes, and block tables before replay; fall back on mismatch |
| Weight, activation, and KV allocators | Per-class allocation policies | Sized during planning, not per launch; KV policy is paged |
| Shape signature and dispatch selectors | Dispatch and tuning lookup | Keyed by dtype, shape class, and row class; values come from measured tables for the target device |
| Backend capabilities | Capability negotiation and unsupported-operation fallback | Consulted during planning; unsupported combinations fail explicitly |

## 7. Memory model

### 7.1 UMA is not a free pass

The APU shares physical memory between CPU and GPU, but allocation paths are not
equivalent. The implementation must measure, for the target system:

- device allocations;
- host pinned allocations;
- mapped/host-visible allocations;
- any available unified-memory mechanisms;
- bandwidth, prefetch behavior, page migration, and synchronization cost.

The allocation policy per buffer class is chosen from those measurements, not
from the assumption that zero-copy access is optimal.

### 7.2 Pools

| Pool | Contents | Lifetime | Policy requirement |
| --- | --- | --- | --- |
| **Weights** | Quantized weights, embedding tables, output head | Model lifetime | Device-resident and uploaded once; addresses stay stable while the model is loaded |
| **Activation** | Layer scratch, attention workspace, logits | Per step or per batch | Reused across layers and sized from the plan; no per-kernel allocation in the hot path |
| **KV cache** | Key/value blocks | Request lifetime, with prefix-cache sharing | Paged block allocator with F32, F16, or Q8_0 rows, addressed through a block table; see [serving.md](serving.md) |
| **Staging** | Host/device transfer buffers | Transient | Reused with explicit synchronization; the class of each staging buffer follows section 7.1 |
| **Graph workspace** | Stable-address buffers for replay | Session lifetime | Address stability is required while a captured graph is installed |

### 7.3 KV cache summary

The KV cache is a paged pool addressed by allocator block ID. A request owns a
block table covering its prompt plus output budget; the model code sees only the
logical position, never the physical block layout. Positions are resolved
through the block table by attention and by KV writes. Recurrent convolution and
matrix state are the exception: they stay contiguous per session and are
snapshotted when a prefix is shared.

The storage format is a per-run choice among F32, F16, and Q8_0, using the same
block table and the same kernel entry points. The block size is configurable.
Block allocation, reference counting, prefix sharing, cancellation, and eviction
are specified in [serving.md](serving.md).

## 8. Backends and fallback

| Backend | Role | Requirement |
| --- | --- | --- |
| HIP GPU | Primary execution path | Implements the full prefill and decode graph for every supported architecture; device errors are reported, never suppressed |
| CPU reference | Correctness oracle and required fallback | Implements the same operators and the same KV and recurrent-state interfaces, and is the default when no device path is selected |
| XDNA NPU | Optional heterogeneous acceleration | May report capabilities and fall back to the GPU; must never be a dependency of the GPU-only path; see [npu.md](npu.md) |

Planning validates supported dtypes, quantization types, batch limits, graph
eligibility, and buffer shapes per operation, consulting the backend capability
description. If a kernel or dtype is unsupported, the planner either selects a
measured dispatch path or fails during planning with a precise diagnostic.
Silent numerical substitution is not allowed, and a rejected operation is never
appended to a plan.

Reserved for future extension:

- `Subgraph` and `DeviceAffinity` for multi-device partitioning;
- `Transfer` and `Event` for cross-device dependencies;
- NPU-specific weight formats and model copies.

These interfaces must remain optional so the GPU-only release does not depend on
NPU toolchain availability.

## 9. Key decisions

| Question | Decision |
| --- | --- |
| GPU kernels in pure Go? | No. Go owns the runtime; HIP/C++ owns device computation. |
| Depend on GGML? | No GGML inference engine. Use GGUF documentation and reference implementations only for format and result validation. |
| rocBLAS/rocWMMA? | Initial baseline and fallback; replace hot paths with purpose-built kernels incrementally. |
| General graph compiler? | No in the first release. A lightweight `ExecutionPlan` is sufficient. |
| Develop CPU, GPU, and NPU simultaneously? | GPU first, CPU reference for correctness, NPU later. |
| First model? | Qwen3.5-4B hybrid (`qwen35`), Q4_K_M GGUF. |
| Optimization order | End-to-end loop → quantized GEMV → WMMA quantized GEMM → attention → scheduler. |
| NPU scheduling in the first release? | No. Reserve backend and event abstractions only. |
| When to add HIP Graph? | After the plain stream path is stable and profiling shows launch overhead is significant. |

## 10. Open design questions

These require implementation experiments or explicit decisions. The current
decision is recorded with each question; reopening one requires new
measurements or a scope change recorded in the same change.

1. **Buffer and event ownership syntax.** Buffers are caller-owned opaque C
   handles with explicit release; the Go wrapper makes `Close` idempotent. An
   event represents one submission, and `Wait` may be called more than once;
   implementations must make it idempotent after completion. Plans do not own
   bound buffers, and the design does not require an arena allocator.
2. **Allocation policy per class.** Weights, activations, KV, and recurrent
   state are device allocations; mapped host memory is measured but is not a
   default path. Upload strategy follows the measurements required by section
   7.1. Revisit only with a measured prefetch or migration win.
3. **Graph replay mutation.** Capture is per session for the decode shape.
   Token and position reach the device through a device-resident dynamic
   buffer; replay validates the session, position, and block-table coverage and
   falls back to stream submission on any failure. Replay is opt-in: it is
   enabled only when profiling shows launch overhead is material.
4. **Dispatch decisions.** Measured thresholds are encoded in explicit Go
   selectors keyed by dtype, shape class, and row class, derived from the
   checked-in sweep harness and recorded in the benchmark reports. There is no
   runtime autotuner and no device table registry; adding a device means
   re-running the sweeps and updating the selectors and their tests. Values
   measured on another RDNA part are not accepted.
5. **Fallback semantics.** Unsupported dtype/shape combinations fail during
   planning with a capability error; silent numerical substitution is
   forbidden. A library GEMM may be selected for prefill only through a measured
   decision. Decode stays on fused quantized GEMV.
6. **Prefix-cache eviction.** LRU, with reference-counted blocks: an entry
   releases only its own reference while request-held references keep shared
   blocks alive. Entry and block caps are configurable, and allocator pressure
   invokes the cache before allocation fails.
7. **Mixed-shape batching.** The scheduler interleaves per-request plans on one
   stream. Fusing several decode requests, or prefill plus decode, into one
   kernel batch is not part of the first release: it needs per-lane block tables
   and sequence lengths.
8. **NPU integration boundary.** Future task; not scheduled. The NPU backend
   reports capabilities and a fallback decision only. Execution, graph
   compilation, and subgraph splitting are out of scope until the activation
   criteria in [npu.md](npu.md) are met on the reference host.

## 11. Extensibility rules

- **New architecture:** add a metadata adapter and a model builder. Do not add
  architecture-specific branches to the GGUF reader or to the HIP runtime.
- **New quantization type:** add reader support, a dtype descriptor, dequant,
  GEMV and GEMM kernels, and planner entries. The model layer must not inspect
  raw packed bytes.
- **New backend:** implement the backend, executable, and event contracts,
  declare capabilities, and provide fallback behavior. Do not modify the
  scheduler.
- **New kernel:** add its shape validation to plan construction, its dispatch
  rule to the selectors, correctness tests, and a performance baseline. The
  planner selects it only when its preconditions hold.
- **New API surface:** add it in the transport layer and map it to scheduler
  requests. Do not bypass admission or KV budgeting.
