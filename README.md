# yuuka

`yuuka` is a from-scratch LLM inference runtime for AMD Strix Halo **gfx1151**
(RDNA 3.5) APUs. The runtime is written in Go; performance-critical device code
is written in HIP/C++. It reads GGUF files directly, keeps quantized weights
packed, and does not use llama.cpp, GGML, or Ollama as an execution engine.

The repository currently holds a single-request vertical slice for the Qwen3.5
hybrid decoder (`qwen35`) and its MoE variant (`qwen35moe`), each with a CPU
reference and a HIP path, and a concurrent serving path:

- a validated, mmap-backed GGUF reader with split-shard support;
- Qwen byte-level BPE and SentencePiece tokenizers with Qwen chat rendering;
- a CPU reference decoder used as the correctness oracle;
- a HIP decoder that runs the full Qwen3.5 and Qwen3.6-MoE models on the GPU,
  with rocWMMA/rocBLAS prefill GEMM, an int8 MMQ route for K-quant weights,
  paged Flash Attention, chunked prefill, and on-device IQ2/IQ3 expert decoding;
- a continuous scheduler with paged KV, reference-counted prefix caching, and
  request cancellation, plus opt-in HIP Graph replay;
- a CLI (`inspect`, `run`, `bench`) and an OpenAI-compatible server with SSE
  streaming, health, metrics, and pprof endpoints.

Only `qwen35` and `qwen35moe` are supported, on both the CPU reference and the
HIP backend. Other architectures, unsupported tensor types, and non-Qwen chat
templates are rejected explicitly rather than approximated. See
[`docs/README.md`](docs/README.md) for release scope and the model reports for
measured status and limits.

## Quick start

### 1. Build

CPU-only (no ROCm dependency):

```bash
mkdir -p bin
go build -o bin/yuuka ./cmd/yuuka
go build -o bin/yuuka-server ./cmd/yuuka-server
```

HIP backend (see [Build](#build) for details):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
mkdir -p bin
go build -tags hip -o bin/yuuka-hip ./cmd/yuuka
go build -tags hip -o bin/yuuka-server-hip ./cmd/yuuka-server
```

### 2. Inspect a model

```bash
export MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf
./bin/yuuka inspect --model "$MODEL" --tensors --hash
```

### 3. Generate text

```bash
# CPU reference backend
./bin/yuuka run --model "$MODEL" --prompt "The capital of France is" --max-tokens 32

# HIP backend with chat rendering
./bin/yuuka-hip run \
  --model "$MODEL" --backend hip --chat \
  --system "You are concise." \
  --prompt "Explain RoPE in one sentence." \
  --max-tokens 64
```

HIP binaries still default to the CPU backend; select the GPU with
`--backend hip`. A binary built without the `hip` tag rejects that flag with an
explicit error. `--temperature 0` means greedy sampling.

### 4. Serve an OpenAI-compatible API

```bash
./bin/yuuka-server-hip --model "$MODEL" --backend hip \
  --ctx 4096 --model-name yuuka --addr 127.0.0.1:8080

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"yuuka","messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"temperature":0}'
```

Endpoints: `POST /v1/chat/completions` (JSON or SSE), `GET /v1/models`,
`GET /healthz`, `GET /metrics`, `GET /debug/pprof/*`. Add `"stream": true` for
SSE. Serving limits and request parameters are documented in
[`docs/serving.md`](docs/serving.md).

## Build

### Requirements

| Component | Requirement |
| --- | --- |
| Host OS | Linux for the HIP backend; the CPU-only build has no ROCm dependency |
| Go | 1.24 or newer |
| Native toolchain | C++17 compiler and CMake 3.20 or newer |
| ROCm | HIP 7.16 on the ROCm 10.1 layout, validated on `gfx1151` |
| GPU | AMD `gfx1151` (Radeon 8060S) for the validated HIP path |
| Native libraries | rocBLAS and rocWMMA development packages plus OpenMP support |

CMake searches `ROCM_PATH` and the usual `/opt/rocm` layout.

### Native library and Go binaries

The cgo link path expects the native build directory to be `build/` at the
repository root. Use the `hip` build tag for GPU support.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

mkdir -p bin
go build -tags hip -o bin/yuuka-hip ./cmd/yuuka
go build -tags hip -o bin/yuuka-server-hip ./cmd/yuuka-server
```

The default native build also compiles the C ABI tests, kernel tests, and the
hardware probe. `-DYUUKA_BUILD_BENCH=OFF` skips the probe and sweep binaries,
but `libyuuka_hip` still requires rocBLAS and rocWMMA; use the CPU-only Go build
if those development packages are unavailable.

### Tests

```bash
go build ./...
go test ./...
go test -race ./...
go vet ./...

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

Real-model tests require a model file and are skipped otherwise:

```bash
YUUKA_TEST_MODEL="$MODEL" go test -tags hip ./... -count=1
YUUKA_TEST_MOE_MODEL="$MOE_MODEL" go test ./model/qwen35 -run Qwen35MoE -count=1
```

## Documentation

Start at [`docs/README.md`](docs/README.md). The design contracts are:

| Document | Contents |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | Layers, interfaces, memory ownership, Go/HIP boundary |
| [`docs/gguf.md`](docs/gguf.md) | GGUF parsing, metadata, tokenizer, model support matrix |
| [`docs/kernels.md`](docs/kernels.md) | Prefill/decode kernels, quantization, attention, dispatch |
| [`docs/serving.md`](docs/serving.md) | Scheduler, paged KV, HIP Graph, API, observability |
| [`docs/testing.md`](docs/testing.md) | Correctness strategy, test matrix, benchmark protocol, regression gates |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Benchmark report conventions, required metadata and coverage, and the per-model report index |
| [`docs/hardware.md`](docs/hardware.md) | Measured gfx1151 toolchain, memory, rocBLAS, rocWMMA facts |
| [`docs/npu.md`](docs/npu.md) | Deferred NPU strategy and interface reservations |

## Contributing

Read [`AGENTS.md`](AGENTS.md) first: it is the operating contract for this
repository, covering document precedence, design constraints, phase order, and
the verification bar. Contributions are expected to:

1. **Follow the design documents.** If a contract is missing or contradictory,
   resolve it in the owning document before building on it, and update that
   document in the same change as the code.
2. **Keep correctness first.** Establish tensor-level and token-level
   correctness before optimizing; the CPU reference backend stays available as
   the oracle.
3. **Respect the architecture rules.** GGUF, tokenizer, and tensor packages must
   not import HIP; backends must not parse GGUF or branch on model
   architecture; the C ABI stays opaque and exception-free; unsupported
   operations fail explicitly instead of silently falling back.
4. **Verify before handing off.** Run the build, formatting (`gofmt -w`), and
   the tests that cover the change, plus a smoke run of the actual behavior.
   Report what ran, what passed, and what was skipped. Do not weaken a test to
   make a change pass.
5. **Write English only.** Comments, identifiers, log and error messages,
   documentation, and commit messages must be in English.
6. **Commit atomically.** One logical change per commit, imperative subject,
   with the reason and the tests that were run in the body.

Do not use llama.cpp, GGML, or Ollama as an execution engine, do not expand
quantized weights to FP16 for decode, and do not commit model files, build
outputs, or generated artifacts. Benchmark claims require the model hash,
quantization, ROCm/driver version, device, power profile, sampling settings,
warmup, repetitions, and concurrency described in
[`docs/testing.md`](docs/testing.md).

## License

MIT. See [`LICENSE`](LICENSE).
