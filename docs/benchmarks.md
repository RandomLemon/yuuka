# Benchmark Report: Qwen3.5-4B Q4_K_M

This report records a single-request CPU-versus-HIP comparison. It follows the
metadata requirements in [testing.md](testing.md) as far as the current tooling
allows. It is a correctness-phase baseline, not a tuned performance result.

## Environment

| Field | Value |
| --- | --- |
| Date | 2026-11-11 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf` |
| Model SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Model size | 2,740,937,888 bytes |
| Architecture | `qwen35`, 32 layers, 8 full-attention, 24 gated delta-net |
| Quantization | Q4_K_M, mixed per-tensor dtypes |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| Host | Ryzen AI MAX+ 395, 32 threads, 120 GiB shared memory |
| Kernel | 6.17.13-21-pve |
| Sampling | Greedy, temperature 0 |
| Warmup | 1 request |
| Repetitions | 3 measured requests |
| Context capacity | CPU 88, HIP 96 |
| Prompt | 16 tokens from repeated English prose |
| Decode | CPU 8 tokens, HIP 16 tokens |

The CPU and HIP runs use different decode lengths, so compare throughput and
latency, not total wall time. The prompt construction and sampling settings are
identical.

## Results

| Metric | CPU reference p50 | CPU p95 | HIP p50 | HIP p95 |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 7.260 s | 7.350 s | 1.433 s | 1.433 s |
| Prefill throughput | 2.20 tok/s | — | 11.17 tok/s | — |
| TTFT | 7.260 s | 7.350 s | 1.433 s | 1.433 s |
| Decode | 3.631 s / 8 tokens | 3.644 s | 1.519 s / 16 tokens | 1.521 s |
| Decode throughput | 1.93 tok/s | — | 9.87 tok/s | — |

Effective weight-read bandwidth during decode, estimated as model bytes times
tokens per second:

| Backend | Effective bandwidth |
| --- | ---: |
| CPU reference | 5.3 GB/s |
| HIP | 27.0 GB/s |

This is an effective estimate, not a hardware counter. The HIP GEMV path is
still far below the measured `hipMemset` rate (about 190-230 GB/s across the
recorded runs). The current kernel launches one 256-thread workgroup per output
channel and performs a block-wide reduction per quantization block. Reducing
launch and reduction overhead, increasing per-workgroup output reuse, and
vectorizing packed loads are the obvious next optimizations.

## Correctness evidence

For the prompt `The capital of France is`:

- CPU and HIP greedy tokens are identical for the first six generated tokens:
  `[11751, 13, 198, 32, 13, 2912]`.
- The maximum absolute logit difference at the first generated position is
  `4.58e-5`.
- The checked-in GPU test compares final logits and greedy tokens with the CPU
  reference. The checked-in `TestGPULayerTraceMatchesCPU` compares every layer's
  hidden state for the same prompt; the worst recorded layer difference is
  `5.53e-5`. The logit test enforces the greedy token sequence and a loose `5.0`
  maximum absolute logit guard; `4.58e-5` is the observed first-step difference,
  not the enforced threshold.
- Golden CPU token tests also cover `1+1=` and `Once upon a time`.

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

go build -o /tmp/yuuka-cpu ./cmd/yuuka
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka

/tmp/yuuka-cpu bench \
  --model /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt-tokens 16 --decode-tokens 8 --warmup 1 --reps 3 --hash

/tmp/yuuka-hip bench \
  --model /path/to/Qwen3.5-4B-Q4_K_M.gguf --backend hip \
  --prompt-tokens 16 --decode-tokens 16 --warmup 1 --reps 3 --hash
```

The benchmark tool reports P50 and P95 over the measured repetitions. It does
not yet record power, clock, or thermal state.
