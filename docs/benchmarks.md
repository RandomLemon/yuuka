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

## Per-quantization decode baseline

The Q4_K_M source model was requantized with llama.cpp to Q8_0, Q6_K, and a
Q5_K/Q6_K mix. These files are test variants, not independently quality-tuned
models. Conditions: `gfx1151`, HIP backend, prompt 16 tokens, decode 16 tokens,
greedy sampling, context 128, warmup 1, repetitions 2.

| Model variant | File type | File size | Prefill p50 | Prefill tok/s | Decode p50 | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | `15` | 2,740,937,888 | 1.878 s | 8.52 | 1.731 s | 8.67 |
| Q8_0 | `7` | 4,482,403,488 | 0.807 s | 19.82 | 1.053 s | 14.24 |
| Q6_K | `18` | 3,464,055,968 | 1.528 s | 10.47 | 1.311 s | 11.44 |
| Q5_K/Q6_K mixed | `18` | 3,074,987,168 | 1.572 s | 10.18 | 1.651 s | 9.09 |

SHA-256 values: Q4_K_M `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`,
Q8_0 `dc3021458088a6fb0e8629c096ea7e95f743e434d96dc4351d2107c669be1fa9`,
Q6_K `438c752e79f8cfdf80483b7017ebc672f16eaf717f198c298740eb6caec097e4`,
Q5_K `68060caab98d455ed40683ab44181599d9ac528ec2378f1c10683b8b8c790637`.
The CPU/GPU agreement test passes for all four variants.

Effective weight-read bandwidth, estimated as model bytes times decode tok/s:
Q4_K_M 23.8 GB/s, Q8_0 63.8 GB/s, Q6_K 39.6 GB/s, and the Q5_K mix 28.0 GB/s.
These remain below the measured memset range of about 190-230 GB/s.

## Decode kernel attribution

`TestGPUProfileDecode` runs one ordinary decode step after a fixed prompt with
HIP events around every submitted plan operation. This is event-instrumented
attribution, not a hardware-counter profile; total time includes event overhead.

| Operation | Time | Share |
| --- | ---: | ---: |
| GEMV | 92.859 ms | 90.5% |
| GDN | 7.308 ms | 7.1% |
| RMSNorm | 0.807 ms | 0.8% |
| residual add | 0.374 ms | 0.4% |
| L2Norm | 0.258 ms | 0.3% |
| gated RMSNorm | 0.237 ms | 0.2% |
| conv1d | 0.174 ms | 0.2% |
| SwiGLU | 0.161 ms | 0.2% |
| attention | 0.139 ms | 0.1% |
| head RMSNorm | 0.096 ms | 0.1% |

The profiled step submitted 579 operations and took 102.580 ms. The GEMV path is
the clear decode bottleneck, consistent with the effective bandwidth remaining
far below device copy rates.

## Batched GEMV dispatch

`build/yuuka_gemv_sweep` compares one GEMV per batch row against the batched
GEMV kernel for `n_in = n_out = 2560`, two warmups, ten repetitions, on gfx1151.
Values are the batched speedup over per-row submission; host timing includes the
completion wait for both paths.

| Dtype | batch 1 | batch 2 | batch 4 | batch 8 | batch 16 | Batched path from |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q4_K | 0.98 | 1.58 | 2.34 | 3.20 | 3.88 | batch 2 |
| Q5_K | 1.02 | 1.71 | 2.57 | 3.55 | 4.22 | batch 2 |
| Q6_K | 1.02 | 1.25 | 1.90 | 2.49 | 3.16 | batch 2 |
| Q8_0 | 0.61 | 0.86 | 1.26 | 1.59 | 1.71 | batch 4 |
| F16 | 0.81 | 0.94 | 1.22 | 1.37 | 1.51 | batch 4 |

`backend/hip.SelectGemvPath` encodes these thresholds; batches above 16 fall
back to per-row submission. The current single-token decode path always uses
batch 1, so the batched path is a prefill/batching building block.
`tests/hip/gemv_test.cpp` checks the batched kernel against the reference for
F32, F16, Q8_0, Q4_K, Q5_K, and Q6_K at batch sizes 1, 2, 3, 4, 8, and 16.

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
