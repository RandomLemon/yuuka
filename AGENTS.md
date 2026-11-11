# AGENTS.md

## 1. Purpose

This file is the operating contract for coding agents working on `yuuka`.
Read it before changing code. It defines document precedence, implementation
order, engineering rules, verification requirements, and the English-only
policy for code comments and commit messages.

The design documents in `docs/` are the source of truth for behavior. This file
is the source of truth for how work is performed. If the two conflict, resolve
the conflict explicitly and update the affected document; do not leave code and
documentation inconsistent.

## 2. Project summary

`yuuka` is a from-scratch LLM inference framework for AMD **gfx1151**
(RDNA 3.5) APUs. The runtime is Go. Performance-critical device code is
HIP/C++. It loads GGUF models directly.

`llama.cpp`, GGML, and Ollama are **reference implementations for format,
behavior, and numerical comparison only**. They must not become the inference
execution kernel, a required runtime dependency, or a source of copied
implementation code without license review and attribution.

First target:

> Linux + ROCm + gfx1151 + Qwen3.5 hybrid + Q4_K GGUF + single-request,
> GPU-only inference.

## 3. Current repository state

The repository contains a working single-request vertical slice:

- Go module and package layout for GGUF, tokenizer, tensor, model, engine,
  backends, CLI, and server.
- Pure Go GGUF reader with validation and mmap-backed tensor storage.
- Qwen3.5 (`qwen35`) tokenizer, chat rendering, CPU reference forward pass, and
  golden token tests.
- HIP C ABI, plan executor, and kernels for the full `qwen35` decode graph.
- rocWMMA/rocBLAS prefill GEMM, causal GQA Flash Attention, and chunked prefill.
- `yuuka` CLI and OpenAI-compatible `yuuka-server`.
- Native CTest coverage for the ABI, quantized GEMV, and prefill kernels.

Not yet implemented: paged KV, continuous batching, prefix cache, HIP Graph, and
HIP execution for architectures other than `qwen35`.

Therefore:

- `go build ./...`, `go test ./...`, and `cmake` are expected to work.
- The HIP Go packages require `-tags hip` and a built `libyuuka_hip.so`.
- Do not claim later-phase behavior or performance that has no checked-in
  evidence.
- The default starting point for new work is the first incomplete item in the
  roadmap, unless the user assigns a specific slice.

## 4. Source-of-truth documents

| Area | Document |
| --- | --- |
| System structure, interfaces, Go/HIP boundary, memory model | [docs/architecture.md](docs/architecture.md) |
| GGUF format, metadata, tokenizer, model/quantization support | [docs/gguf.md](docs/gguf.md) |
| Prefill/decode kernels, quantization, attention, dispatch, numerics | [docs/kernels.md](docs/kernels.md) |
| Scheduler, KV cache, HIP Graph, API, observability | [docs/serving.md](docs/serving.md) |
| Correctness strategy, test matrix, benchmarks, regression gates | [docs/testing.md](docs/testing.md) |
| Deferred NPU strategy and interface reservations | [docs/npu.md](docs/npu.md) |
| Phases, deliverables, exit criteria, risks | [docs/roadmap.md](docs/roadmap.md) |
| Documentation index, goals, non-goals, glossary | [docs/README.md](docs/README.md) |

Precedence:

1. For a concrete API or behavioral contract, the document for that area wins.
2. For implementation order and acceptance, `docs/roadmap.md` and
   `docs/testing.md` win.
3. For process and language policy, this file wins.
4. If a contract is missing or contradictory, stop and resolve it in the
   relevant document before building on it.

Update documentation in the same change whenever behavior, interfaces,
limitations, or acceptance criteria change.

## 5. Non-negotiable design constraints

These are requirements, not suggestions:

1. **Correctness before performance.** Establish tensor-level and token-level
   correctness before optimizing a kernel.
2. **Keep quantized weights quantized.** Do not expand all quantized weights to
   FP16 for decode. Fuse dequantization and GEMV.
3. **Prefill and decode are separate operator families.** Do not route decode
   through a large-M GEMM path by default.
4. **The CPU reference backend stays available.** It is the primary correctness
   oracle and the fallback for unsupported operations.
5. **Go does not issue one cgo call per tensor operation.** Build and cache an
   `ExecutionPlan`; submit work in batches or steps.
6. **The C ABI is stable and opaque.** Do not expose C++ objects, STL types, or
   exceptions to Go.
7. **No GPU work may dereference unpinned Go memory.** Persistent data belongs
   in backend-managed buffers; lifetimes are coordinated with events.
8. **Unsupported operations fail explicitly.** Silent numerical substitution,
   silent fallback, or silent truncation is forbidden.
9. **Measure on gfx1151.** Do not port MFMA assumptions, occupancy numbers, or
   performance claims from CDNA hardware.
10. **Do not add out-of-scope systems.** No training, distributed inference,
    multi-GPU tensor parallelism, general graph compiler, or NPU dependency in
    the first release.

## 6. Target platform

| Item | Requirement |
| --- | --- |
| OS | Linux |
| GPU target | gfx1151 / RDNA 3.5 |
| Reference device | Ryzen AI Max+ 395 / Radeon 8060S |
| Go | Current stable release supported by the project |
| HIP | ROCm-supported HIP toolchain; pin and document the version |
| Build | Go Modules for Go; CMake for the HIP shared library |
| First model | Qwen3.5-4B hybrid (`qwen35`), Q4_K_M GGUF |
| First execution mode | Single request, GPU-only |

Hardware and toolchain facts must be verified on the actual machine. Do not
assume GPU-visible memory, mappable host memory, rocWMMA support, or NPU runtime
availability from specifications.

## 7. Repository layout

Create and maintain this structure as implementation proceeds:

```text
yuuka/
├── cmd/
│   ├── yuuka/                 # CLI
│   └── yuuka-server/          # OpenAI-compatible server
├── gguf/                      # Pure Go GGUF reader
├── tokenizer/                 # BPE, SentencePiece, chat templates
├── model/                     # Model interfaces and architecture builders
│   ├── llama/
│   └── qwen3/
├── tensor/                    # Tensor descriptors, dtypes, layouts
├── engine/                    # Planner, executor, sampling
├── scheduler/                 # Admission, batching, requests
├── memory/                    # Allocators, KV cache, prefix cache
├── backend/
│   ├── cpu/                   # Reference backend
│   ├── hip/                   # HIP runtime binding and bridge
│   └── npu/                   # Reserved; no first-release dependency
├── kernels/hip/               # Device kernels by operator family
├── bench/                     # Benchmark harnesses
├── tests/                     # Integration and golden tests
├── docs/                      # Design and roadmap documents
├── scripts/                   # Toolchain, fixture, and CI helpers
├── CMakeLists.txt
└── go.mod
```

Dependency direction:

- `gguf`, `tokenizer`, `tensor`, and model definitions must not import HIP.
- `backend/*` must not parse GGUF or contain model architecture branches.
- `scheduler` must not call device kernels directly.
- Kernels must not own request lifecycle or model metadata.

## 8. Phase discipline

Implement in roadmap order. A phase may be explored early, but it must not be
declared complete before its exit criteria are met.

| Phase | Focus | Do not claim complete without |
| --- | --- | --- |
| **0** | Toolchain, hardware characterization, scaffold, C ABI smoke test | HIP/rocBLAS/WMMA checks and allocation/bandwidth measurements |
| **1** | GGUF, tokenizer, dense model, CPU reference, HIP FP16 baseline | Token-exact or documented logits match; correct stop behavior |
| **2** | Native Q4_K/Q6_K/Q8_0 decode | Correct Qwen3 Q4_K generation; measured decode bottleneck |
| **3** | WMMA quantized GEMM, Flash Attention, chunked prefill | Prompt-length performance curves and same-condition comparison |
| **4** | Paged KV, continuous batching, prefix cache, HIP Graph, API | Reproducible latency/throughput; cancellation and leak checks |
| **5** | MoE, KV quantization, speculative decoding, NPU experiments | Opt-in features with measured benefit and no regressions |

Rules:

- Build the smallest end-to-end vertical slice first.
- One correct single-request path is worth more than a broad set of unfinished
  features.
- Do not skip Phase 1 correctness to make Phase 2 look fast.
- Do not introduce a concurrency feature before the single-request path is
  stable and measured.
- Do not add NPU paths until the GPU-only path is accepted and the criteria in
  [docs/npu.md](docs/npu.md) are met.

## 9. Go conventions

### 9.1 Style and structure

- Run `gofmt` on all Go files. Run `go vet ./...` before considering work done.
- Keep packages focused. Do not create import cycles or a shared `util` package
  for unrelated helpers.
- Define interfaces at the consumer when that improves testability. Do not
  create interfaces solely for mocking trivial data holders.
- Exported identifiers need doc comments that state behavior, ownership, and
  error semantics.
- Prefer explicit structs and functions over reflection, global registries, or
  hidden initialization.
- Keep the hot path allocation-free where practical; avoid per-token map
  lookups, formatting, and interface churn.

### 9.2 Errors

- Return errors; do not panic in library code.
- Wrap errors with context: `fmt.Errorf("load tensor %q: %w", name, err)`.
- Use typed or sentinel errors for capability, validation, and resource
  failures so callers can distinguish them without string matching.
- Never ignore an error from allocation, submission, synchronization, or
  cleanup.
- Cleanup errors must not hide the primary failure; join or report both when
  the language version supports it.
- Errors crossing the C ABI use status codes plus a retrievable message. No C++
  exceptions or Go panics may cross the boundary.

### 9.3 Lifetime and concurrency

- Every buffer has one explicit owner. Document when ownership transfers.
- The executor must not reuse scratch or KV storage before its completion event
  has been observed.
- Use `context.Context` for request cancellation and deadlines. Do not detach
  work from cancellation.
- Do not start a goroutine without a defined exit condition. Tests must detect
  goroutine leaks in request paths.
- Protect shared scheduler and cache state with clear locking. Do not hold a
  lock while waiting on the GPU.
- Run race-enabled tests for CPU-side concurrency: `go test -race ./...`.

### 9.4 cgo and the HIP bridge

- Keep cgo in the backend bridge package only.
- Do not pass Go pointers to asynchronous GPU work.
- Use `runtime.KeepAlive` where a Go object must outlive a synchronous C call.
- Convert C strings explicitly and free returned buffers according to the ABI
  contract.
- Cache handles and plans; do not perform cgo calls per tensor per token.
- Check both the status code and the error message path on every bridge call.

## 10. HIP and C++ conventions

### 10.1 ABI and ownership

- Export a versioned C ABI from a standalone shared library.
- Use opaque handles, plain C types, and exported status codes.
- Do not expose C++ classes, templates, references, or STL containers.
- No exception may cross the C boundary. Catch internally and convert to a
  status code with context.
- Resource release must be idempotent or explicitly reject double close.
- Document whether returned handles are owned by the caller or the library.

### 10.2 HIP runtime and kernels

- Check every HIP API return value. Do not ignore launch or synchronization
  errors.
- Use explicit streams; do not rely on the default stream in library code.
- Do not insert hidden device synchronizations in the hot path.
- Allocate scratch during planning or reuse, not per kernel launch.
- Pass launch dimensions and tile parameters from measured dispatch decisions.
- Avoid device-side `printf` and debug logging in committed kernels.
- Prefer `__launch_bounds__` and resource annotations only when supported by
  measurement.
- Target Wave32 and RDNA WMMA. Do not copy CDNA MFMA kernels.
- Keep kernel code free of model-specific branches. Shape and quantization
  specialization belongs in dispatch.

### 10.3 Numerics

- Accumulate FP16/BF16 matrix products in FP32 unless a documented experiment
  proves another choice within tolerance.
- Apply quantization scales and mins before final rounding; avoid double
  rounding.
- Use stable softmax and normalization implementations.
- Do not enable fast-math or unsafe floating-point contraction unless a
  numerical test demonstrates it stays within the documented tolerance.
- Define whether each kernel is deterministic. Split reductions and atomics
  must be tested for run-to-run variation.
- Document tolerances per stage in the test that enforces them.

### 10.4 Build

- Use CMake for the HIP shared library. Keep the C ABI boundary between the Go
  and C++ build systems.
- Pin and document the ROCm/HIP version used for validation.
- Enable compiler warnings and treat new warnings in touched code as defects.
- Do not commit build directories, compiled shared libraries, or generated
  binaries.

## 11. Build, test, and verification

Once the scaffold exists, these are the canonical commands. Keep them working:

```bash
# Go
go build ./...
go test ./...
go test -race ./...

# HIP library and C tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure

# Formatting
gofmt -w $(find . -name '*.go' -not -path './vendor/*')
```

Add narrower commands for fast iteration, but do not replace the full checks in
CI or in a completion claim.

### 11.1 Verification expectations

Before yielding significant work:

- Run the test that covers the changed behavior.
- Run a smoke path that exercises the change end-to-end.
- Run formatting, build, and relevant tests.
- Report exactly what ran, what passed, and what was skipped.
- If a GPU is unavailable, skip with an explicit reason. Never let a GPU test
  silently pass.
- A test that only proves a function was called, a mock echoed input, or a
  struct field was copied is not sufficient evidence.
- Do not weaken, delete, or re-pin a test to accommodate a regression. Fix the
  implementation or explain why the expected behavior changed.
- Remove obsolete tests and dead code when a contract is intentionally changed.

### 11.2 Benchmarking

- Follow [docs/testing.md](docs/testing.md).
- Record model hash, quantization, ROCm/driver version, device, power profile,
  sampling settings, warmup, repetitions, and concurrency.
- Distinguish prefill from decode, and single-request latency from aggregate
  throughput.
- Do not place performance claims in a commit message or summary without the
  environment and measurement conditions.
- Do not hard-code benchmark results into dispatch decisions without the
  measured table and version metadata.

## 12. Documentation rules

- All documentation is written in English.
- Keep `docs/README.md` as the document index. Add new documents there.
- Do not duplicate a design contract across documents. Link to the owning
  document instead.
- When changing an interface, memory-ownership rule, dispatch policy, API
  behavior, or acceptance criterion, update the owning document in the same
  change.
- Record unresolved decisions in the relevant document's open-questions or
  limitations section. Do not leave them only in code comments.
- Keep diagrams textual or Mermaid-compatible and readable in a terminal.
- Do not claim implementation status that the repository does not have.

## 13. Language policy

**All code comments, doc comments, identifiers, log messages, error messages,
documentation, commit messages, branch names, and pull-request descriptions
must be written in English.**

This applies to:

- Go comments and package documentation;
- HIP/C++ comments, including kernel comments and TODO notes;
- CMake comments and build-script comments;
- test names and test failure messages;
- commit subjects and bodies;
- code review responses recorded in the repository.

Do not introduce comments or commit messages in Chinese or any other language.
If an external reference is in another language, summarize the relevant fact in
English rather than copying the text.

## 14. Git and commit rules

- Make atomic commits: one logical change per commit.
- Do not mix unrelated refactors with behavior changes.
- Commit messages use English and an imperative subject. Recommended form:

```text
<area>: <imperative summary>

<why the change is needed, important constraints, and tests run>
```

Examples:

```text
gguf: validate tensor byte ranges

Add overflow-safe element counts and shard bounds checks so malformed
files fail before device allocation. Tests: go test ./gguf/...
```

```text
kernels: add fused q4_k gemv baseline

Decode reads packed Q4_K weights directly and accumulates in FP32.
Correctness is checked against the CPU reference on aligned and
non-aligned shapes.
```

- Reference the affected document or phase when the change implements a
  roadmap item.
- Do not commit model weights, benchmark datasets, secrets, build outputs, or
  large generated artifacts.
- Do not rewrite shared history or force-push unless the user explicitly asks.
- Run the relevant checks before committing. If a check cannot run, state why in
  the commit body or handoff.
- Respect the repository's MIT license and the licenses of any reference
  implementation. Do not copy upstream code without review and attribution.

## 15. Definition of done

A task is done only when all applicable items are true:

- [ ] The behavior matches the owning design document and the requested scope.
- [ ] The implementation is complete; no stub, placeholder, no-op, or fake
      fallback is presented as finished work.
- [ ] The changed path has a correctness test at the appropriate layer.
- [ ] A smoke run exercised the actual behavior, not only compilation.
- [ ] Relevant build, formatting, and test commands were run and reported.
- [ ] Unsupported cases fail with a clear error or a documented fallback.
- [ ] Resource ownership, cancellation, and cleanup are handled.
- [ ] Documentation and benchmark metadata are updated when contracts change.
- [ ] All comments and commit text are in English.
- [ ] No unrelated files, generated artifacts, or secrets are included.

If blocked, state the exact missing prerequisite, what was tried, and what
remains reachable. Do not silently reduce scope.

## 16. Forbidden actions

- Using llama.cpp, GGML, or Ollama as the inference execution engine.
- Expanding all quantized model weights to FP16 for decode.
- Adding a silent numerical fallback or suppressing a device error.
- Treating a stub, mock, or `TODO` as completed functionality.
- Weakening or deleting a test to make a failing change pass.
- Changing a documented tolerance without evidence and review.
- Claiming performance results without the required benchmark metadata.
- Adding NPU, distributed, multi-GPU, training, or general graph-compiler work
  before the roadmap phase permits it.
- Holding a scheduler lock while waiting on the GPU.
- Passing Go pointers into asynchronous HIP work.
- Committing model files, build outputs, credentials, or generated logs.
- Writing code comments or commit messages in any language other than English.

## 17. Recommended first implementation sequence

For a new implementation agent starting from the current empty state:

1. Produce the Phase 0 hardware/toolchain report.
2. Create the Go module and CMake HIP library scaffold.
3. Implement a minimal C ABI round trip: allocate, submit, wait, release, and
   propagate an error.
4. Add the CPU reference backend skeleton and GGUF parser fixtures.
5. Implement the GGUF reader and tokenizer with golden tests.
6. Implement the CPU reference operators needed by Qwen3.5 hybrid attention and gated delta-net layers.
7. Add the HIP FP16 baseline and the single-request execution plan.
8. Validate logits and greedy tokens against the reference before starting
   quantized kernels.

Keep the first vertical slice narrow. A correct single-request path with
reproducible evidence is the prerequisite for every later phase.
