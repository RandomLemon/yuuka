# yuuka

`yuuka` is a from-scratch LLM inference framework for AMD **gfx1151** (RDNA 3.5)
APUs. The runtime is Go; performance-critical device code is HIP/C++. It reads
GGUF directly and keeps quantized weights packed.

The current vertical slice runs **Qwen3.5-4B** (`qwen35`) on both a CPU
reference backend and a HIP device backend. The HIP path is a planned operation
stream: weights are uploaded once, and one plan submission executes a whole
token.

## Status

| Area | State |
| --- | --- |
| GGUF reader | Validated metadata, tensor directory, split shards, alignment, byte ranges |
| Tokenizer | Qwen3.5 byte-level BPE, special tokens, Qwen chat rendering, golden token IDs |
| CPU reference | Full `qwen35` forward pass, FP32 accumulation, FP32 KV cache |
| HIP execution | Embedding, RMSNorm, RoPE, GQA attention, GDN convolution/state, SwiGLU, LM head |
| Quantized kernels | `F16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K` GEMV, validated against CPU |
| CLI | `inspect`, `run`, `bench` with CPU/HIP selection |
| Server | OpenAI-compatible chat completions, SSE, stop sequences, health, metrics |
| Prefill | Per-token execution; no WMMA/rocBLAS batched prefill yet |
| Serving | Single request; no paged KV, batching, prefix cache, or HIP Graph |

Measured baseline: CPU 1.93 decode tok/s, HIP 9.87 decode tok/s for
Qwen3.5-4B Q4_K_M at short context. See [docs/benchmarks.md](docs/benchmarks.md)
and [docs/hardware.md](docs/hardware.md).

## Requirements

- Linux with ROCm/HIP 7.16 (validated on ROCm 10.1 layout)
- Go 1.24+
- CMake 3.31+
- rocBLAS and rocWMMA development packages for the hardware probe

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

go build -o bin/yuuka ./cmd/yuuka
go build -tags hip -o bin/yuuka-hip ./cmd/yuuka
go build -tags hip -o bin/yuuka-server ./cmd/yuuka-server
```

The `hip` build tag is required for Go code that links the native library.
The default build is CPU-only and does not require `libyuuka_hip.so`.

## Run

```bash
MODEL=~/llm/models/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf

./bin/yuuka inspect --model "$MODEL"
./bin/yuuka run --model "$MODEL" --prompt "The capital of France is" --max-tokens 16
./bin/yuuka-hip run --model "$MODEL" --backend hip --prompt "The capital of France is" --max-tokens 16
```

Chat mode applies the GGUF Qwen chat template:

```bash
./bin/yuuka-hip run --model "$MODEL" --backend hip --chat \
  --system "You are concise." --prompt "Explain RoPE in one sentence." --max-tokens 64
```

## Server

```bash
./bin/yuuka-server --model "$MODEL" --backend hip --ctx 4096 --addr 127.0.0.1:8080

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"stream":false}'
```

Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /healthz`,
`GET /metrics`.

## Tests

```bash
go build ./...
go test ./...
go test -race ./...
go build -tags hip ./...

cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure

# Real-model verification (skipped when the variable is absent)
YUUKA_TEST_MODEL="$MODEL" go test -tags hip ./... -count=1
```

The GPU runtime is checked against the CPU reference on prompt logits and
greedy token sequences. Native tests cover the C ABI and every quantized GEMV
kernel.

## Limitations

- Only `general.architecture = qwen35` is supported.
- Only GPT-2-style byte-level BPE (Qwen2/Qwen3.5 pre-tokenization) is
  implemented; SentencePiece vocabularies are not.
- The HIP executor submits one token at a time and does not batch prefill.
- KV cache is contiguous FP32; paged KV and quantized KV are not implemented.
- The server handles one request at a time and returns `503` when busy.
- Tool and vision chat content is rejected explicitly.

## Documentation

Start at [docs/README.md](docs/README.md). The design contracts are in
[docs/architecture.md](docs/architecture.md), [docs/gguf.md](docs/gguf.md),
[docs/kernels.md](docs/kernels.md), [docs/serving.md](docs/serving.md),
[docs/testing.md](docs/testing.md), and
[docs/roadmap.md](docs/roadmap.md).
