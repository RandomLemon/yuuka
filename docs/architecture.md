# Architecture

## 1. Purpose and scope

This document describes the intended structure of `yuuka`: a GGUF inference
framework with a Go runtime and HIP device kernels, targeting a single AMD
gfx1151 APU. It covers the layer boundaries, the core interfaces, the Go/HIP
seam, memory ownership, backend selection, and the repository layout.

Kernel algorithms are specified in [kernels.md](kernels.md); request scheduling
and serving are specified in [serving.md](serving.md); format handling is
specified in [gguf.md](gguf.md).

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
Request Scheduler / Admission / Continuous Batching (Go)
                   |
Model Executor / Prefill / Decode / Paged KV (Go)
                   |
Execution Planner / Operator Dispatch / Memory Planner (Go)
                   |
             Backend Interface
           /         |          \
      HIP GPU     CPU Ref.     XDNA NPU (future)
           |
   HIP Streams / Events / Graph / Allocators
           |
   Quant GEMV | Quant GEMM | Flash/Paged Attention
   Norm | RoPE | Activation | Sampling | Fusion
```

### 3.1 Layer responsibilities

| Layer | Owns | Must not own |
| --- | --- | --- |
| API and serving | HTTP/CLI, request parsing, SSE streaming, health and metrics | Model semantics, device memory layout |
| Model and tokenizer | GGUF metadata interpretation, tokenizer, chat template, sampling policy | HIP calls, request scheduling |
| Scheduler | Admission, batching, token/KV budgets, cancellation | Kernel selection, weight layout |
| Executor | Prefill/decode orchestration, KV block tables, sampling loop | Format parsing, transport concerns |
| Planner | `ExecutionPlan`, operator selection, scratch sizing, kernel signatures | Device-specific kernel internals |
| Backend | Device allocation, streams, events, kernel launch, graph capture | GGUF parsing, model architecture decisions |
| Kernels | Device-side computation | Model metadata, request lifecycle |

**Dependency rule:** higher layers depend on interfaces, not on HIP. A
replacement kernel or backend must not require changes to the GGUF reader or
model definitions.

## 4. Core interfaces

The following interfaces define the intended seams. They are a design shape,
not a frozen API.

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
    Alloc(size uint64) (Buffer, error)
    Compile(plan *ExecutionPlan) (Executable, error)
    Synchronize() error
}

type Executable interface {
    Run(batch *Batch) (Event, error)
    Close() error
}

type Event interface {
    Wait() error
}
```

Implementation status: `backend/backend.go` remains the intended device seam
and defines `Backend`, `ExecutionPlan`, `Batch`, and related types, but the
implemented execution path uses narrower concrete interfaces:

- `serving.Backend` and `serving.Sequence` for request construction and
  prefill/decode steps;
- `scheduler.Task` for admission-driven request state;
- `engine.Runtime` for the single-request CLI;
- `engine/gpu.Model` and `engine/gpu.Session` for the shared HIP model and
  per-request paged KV state;
- `backend/hip.Plan`, `Graph`, `Stream`, `Event`, and `Buffer` at the C ABI.

The generic `backend.Backend` interface can be adopted later, but it must not
introduce a cgo call per tensor and must preserve the current block-table and
session ownership rules.

### 4.1 Interface contracts

These contracts must be explicit in the implementation:

- **Buffer ownership.** Every `Buffer` has one owner. Transfer of ownership and
  the point of release are part of the API, not an implementation detail.
- **Event lifetime.** An event represents completion of submitted work. A
  buffer or scratch region may be reused only after the corresponding event has
  completed. Whether `Wait` is idempotent or single-use must be specified.
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

`BuildPlan` inspects the model, backend capabilities, quantization types, and
expected batch shapes. It produces an `ExecutionPlan` containing:

- weight buffer bindings and device residency;
- kernel choices and their shape signatures;
- scratch and workspace sizes;
- stream and event structure;
- graph-capture candidates, if supported;
- fallback choices for unsupported operations.

Planning happens at load time and whenever a new shape class requires a new
kernel configuration. It must not perform a cgo call per tensor per token. Each
HIP session caches one prefill plan per distinct chunk length; the final partial
chunk has its own plan. A model-level prefill-generation counter invalidates a
session's cached chunk plans when the chunk geometry changes.

## 5. Execution model

```text
Open and validate GGUF
        |
Build tokenizer, model config, weight descriptors
        |
Build ExecutionPlan (kernel selection, scratch, residency)
        |
Allocate weights, activation pool, KV pool, graph workspace
        |
Prefill: prompt chunks -> attention + GEMM -> KV writes
        |
Decode loop: quantized GEMV + paged attention + sampling
        |
Stream tokens / finish / release request resources
```

Current status: both CPU and HIP `qwen35` paths process prompts through cached
chunked prefill plans (batched WMMA or rocBLAS GEMM, paged Flash Attention, and
paged KV writes) and then decode through the single-token paged plan. The
serving scheduler interleaves requests on one model stream. HIP Graph capture
and replay are implemented per session but opt-in; the plain stream path is the
default because the measured decode gain was within noise.

Key rules:

1. GGUF data is validated before any device allocation.
2. Weights are loaded into a device-resident pool for the model lifetime.
3. Prefill and decode share the KV cache but use different operator paths.
   Prefill uses chunked batched plans; decode uses the single-token GEMV and
   per-head attention graph. Dense models still use the decode graph for
   prompts because their HIP execution path is not implemented.
4. Sampling may run on the GPU or on the host through a low-overhead path; the
   choice is a plan decision, not a per-token branch in model code.
5. Request cancellation releases KV blocks and scratch through the scheduler
   and memory manager, not through ad hoc frees in the API layer. The
   single-request runtime retains its own table for reset; the serving path
   releases each request table exactly once.

## 6. Go and HIP boundary

Go must not make a fine-grained cgo call for every tensor operation. The
boundary is designed around planned, batched submission:

```text
Go builds/caches ExecutionPlan
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
- Prefer persistent data in HIP-managed buffers, with events coordinating reuse
  and release.
- Host/device transfers use explicit staging buffers where required.
- Every path that can fail after partial allocation must have a cleanup path
  that releases all acquired buffers and streams.

### 6.3 Required abstractions

| Abstraction | Purpose |
| --- | --- |
| `Device`, `Stream`, `Event` | Execution and synchronization |
| `DeviceBuffer`, `HostPinnedBuffer`, `MappedBuffer` | Allocation classes with different performance and lifetime rules |
| `GraphExecutable`, `GraphReplay` | Cached launch graphs for stable shapes |
| `ScratchAllocator`, `WeightAllocator`, `KVAllocator` | Pool-specific allocation policies |
| `KernelRegistry`, shape signature | Dispatch and tuning lookup |
| Backend capabilities | Capability negotiation and unsupported-operation fallback |

## 7. Memory model

### 7.1 UMA is not a free pass

The APU shares physical memory between CPU and GPU, but allocation paths are not
equivalent. The implementation must measure, for the target system:

- `hipMalloc` device allocations;
- host pinned allocations;
- mapped/host-visible allocations;
- any available unified-memory mechanisms;
- bandwidth, prefetch behavior, page migration, and synchronization cost.

The memory planner chooses a policy per buffer class based on measurements, not
on the assumption that zero-copy access is optimal.

### 7.2 Pools

| Pool | Contents | Lifetime | Allocation policy |
| --- | --- | --- | --- |
| **Weights** | Quantized weights, embedding tables, output head | Model lifetime | Device-resident; stable addresses preferred |
| **Activation** | Layer scratch, attention workspace, logits | Per step or per batch | Reused across layers; sized from the plan |
| **KV cache** | Key/value blocks | Request lifetime, with prefix-cache sharing | Paged `F32` block allocator; immutable per-request block table; see [serving.md](serving.md) |
| **Staging** | Host/device transfer buffers | Transient | Reused; explicit synchronization |
| **Graph workspace** | Stable-address buffers for replay | Session lifetime | Address stability is required while a captured graph is installed |

### 7.3 KV cache summary

Current implementation: CPU and HIP use paged `F32` KV pools addressed by
allocator block ID. A request owns a block table covering its prompt plus output
budget; positions are resolved through that table by the CPU attention loop and
by the paged HIP KV-write and attention kernels. Recurrent convolution and
matrix state stay contiguous per session and are snapshotted for prefix reuse.
The default block size is 16 tokens. FP16 KV and quantized KV remain extension
points. The logical KV view exposed to model code does not expose physical block
layout. Block allocation, reference counting, prefix sharing, cancellation, and
eviction are specified in [serving.md](serving.md).

## 8. Backends and fallback

| Backend | Role | Status |
| --- | --- | --- |
| HIP GPU | Primary execution path | Implemented for `qwen35` single and concurrent requests over a shared paged KV pool; opt-in graph replay |
| CPU reference | Correctness baseline, required fallback for unsupported operations | Implemented for `qwen35`; default backend; concurrent paged serving |
| XDNA NPU | Optional heterogeneous acceleration | Deferred; see [npu.md](npu.md) |

The planner queries backend capabilities, including supported dtypes,
quantization types, maximum batch, graph support, and memory properties. If a
kernel or dtype is unsupported, the planner must either select a supported
fallback or fail during planning with a precise diagnostic. Silent numerical
substitution is not allowed.

Plan construction applies the same rule to every GEMV, batched GEMV, and GEMM
operation. An unsupported dtype or a width that violates the quantization
block layout is a capability error, and invalid dimensions, batch sizes, or
buffer ranges are argument errors. A rejected operation is not appended to the
plan, so execution cannot silently skip it.

Reserved for future extension:

- `Subgraph` and `DeviceAffinity` for multi-device partitioning;
- `Transfer` and `Event` for cross-device dependencies;
- NPU-specific weight formats and model copies.

These interfaces must remain optional so the first GPU-only release does not
depend on NPU toolchain availability.

## 9. Repository layout

Current implemented layout:

```text
yuuka/
├── cmd/
│   ├── yuuka/                 # CLI
│   └── yuuka-server/          # OpenAI-compatible server
├── gguf/                      # Pure Go GGUF reader and validation
├── tokenizer/                 # Qwen BPE, Llama SentencePiece, and chat rendering
├── tensor/                    # Dtypes, block layouts, CPU reference math
├── model/
│   ├── dense/                 # Llama and Qwen3 CPU reference decoders
│   └── qwen35/                # Config, weights, CPU reference decoder, state
├── engine/
│   ├── engine.go              # Single-request generation loop
│   ├── sampling.go            # Greedy, top-k, nucleus sampling
│   └── gpu/                   # Shared HIP model plus paged sessions
├── memory/                    # Block allocator, block tables, prefix cache
├── scheduler/                 # Admission, continuous batching, timeouts
├── serving/                   # Request tasks, backends, prefix integration
├── backend/
│   ├── backend.go             # Planned backend interfaces; not wired yet
│   └── hip/
│       ├── hip.go             # cgo binding for the C ABI
│       └── bridge/            # HIP C ABI implementation
├── kernels/hip/
│   ├── attention/
│   ├── embedding/
│   ├── gemm/
│   ├── gemv/
│   ├── ops/
│   ├── ssm/
│   ├── common.cuh
│   └── yuuka_kernels.h
├── tests/
│   ├── fixture/               # Deterministic tiny GGUF generators
│   └── hip/                   # C ABI, GEMV, GEMM, operators, paged KV, graph
├── bench/
│   ├── hip/                   # Hardware probe and kernel sweeps
│   └── serving/               # Latency and throughput harness
├── scripts/
├── docs/
├── CMakeLists.txt
└── go.mod
```

Planned but not present yet: a separate `backend/cpu/` package, `backend/npu/`,
separate `model/llama/` and `model/qwen3/` packages, and
`kernels/hip/sampling/`. The CPU reference model code is implemented in
`model/qwen35` and `model/dense`; `engine` owns the single-request generation
loop, while `serving` and `scheduler` own concurrent requests. The generic
`backend` interfaces remain the target shape; the current executor uses
`serving.Backend`, `engine/gpu.Model`, `engine/gpu.Session`, and
`backend/hip.Plan`.

Use CMake for the HIP shared library and Go Modules for Go components. The two
build systems meet at the C ABI. Do not embed the HIP/C++ build inside a Go
package.

## 10. Key decisions

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

## 11. Open design questions

These require implementation experiments or explicit decisions. The current
decision and the phase that may reopen it are recorded below.

1. **Buffer and event ownership syntax.** Buffers are caller-owned opaque C
   handles with explicit release; the Go wrapper makes `Close` idempotent.
   Events are single-use completion signals and `Wait` is valid after the
   signal. Plans do not own bound buffers. Model-lifetime allocations and
   request session allocations are tracked separately; Phase 4 did not need an
   arena allocator.
2. **UMA allocation policy.** Weights, activations, KV, and recurrent state use
   device allocations; transfers stage through pinned host memory. Mapped host
   memory is measured but is not a default path. Revisit only with a measured
   prefetch or migration win.
3. **Graph replay mutation.** Implemented in Phase 4 and opt-in. Capture is per
   session for the decode shape. Token and position are read from a
   device-resident dynamic buffer uploaded before `hipGraphLaunch`; replay
   validates the session, position, block-table coverage, and dynamic buffer
   and falls back to stream submission on any failure. The measured decode gain
   on the validated model was within noise, so the plain stream path is the
   default.
4. **Dispatch tables.** Tables are versioned Go data keyed by operator, dtype,
   shape class, batch class, and measured device. They are generated from the
   checked-in benchmark harness and gated by correctness tests. Values from
   other RDNA parts are not accepted.
5. **Fallback semantics.** Unsupported dtype/shape combinations fail during
   planning with a capability error; silent numerical substitution is forbidden.
   rocBLAS may be selected for prefill only through a measured dispatch table.
   Decode stays on fused quantized GEMV.
6. **Prefix-cache eviction.** LRU is selected for Phase 4. Entries release
   their own references while request-held references keep shared blocks alive.
   Entry and block caps are configurable; allocator pressure invokes the cache
   before allocation fails.
7. **Mixed-shape batching.** The scheduler interleaves per-request plans on one
   stream. Fusing multiple decode requests or prefill plus decode into one
   kernel batch is not implemented; it would require per-lane block tables and
   sequence lengths and is left to a later phase.
8. **NPU integration boundary.** Deferred. NPU execution remains opt-in and must
   fall back to the GPU path; the split is not selected before the Phase 5
   criteria in [npu.md](npu.md) are met.

## 12. Extensibility rules

- **New architecture:** add a metadata adapter and a model builder. Do not add
  architecture-specific branches to the GGUF reader or HIP runtime.
- **New quantization type:** add reader support, a dtype descriptor, dequant/GEMV
  and GEMM kernels, and planner entries. The model layer must not inspect raw
  packed bytes.
- **New backend:** implement `Backend`, `Executable`, and `Event`, declare
  capabilities, and provide fallback behavior. Do not modify the scheduler.
- **New kernel:** register it with a shape signature, correctness tests, and a
  performance baseline. The planner selects it only when its preconditions hold.
- **New API surface:** add it in the transport layer and map it to scheduler
  requests. Do not bypass admission or KV budgeting.
