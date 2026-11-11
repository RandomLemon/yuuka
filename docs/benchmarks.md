# Benchmark Report: Qwen3.5-4B Q4_K_M

This report records single-request CPU-versus-HIP measurements, the Phase 2
decode baseline, and the Phase 3 prefill results. It follows the metadata
requirements in [testing.md](testing.md) as far as the current tooling allows.
The decode section is a correctness-phase baseline, not a tuned performance
result; the Phase 3 section records measured prefill behavior and its known
gap to llama.cpp.

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
- The decode-path maximum absolute logit difference at the first generated
  position is `4.58e-5`. Phase 3 routes multi-token prompts through chunked
  prefill; the same prompt in that path differs from the CPU reference by
  `3.65e-3` in the worst logit and selects the same top-1 token. Both numbers
  are observations under the current kernels, not enforced tolerances except
  for the loose `5.0` logit guard in the GPU/CPU test.
- The checked-in GPU test compares final logits and greedy tokens with the CPU
  reference. The checked-in `TestGPULayerTraceMatchesCPU` compares every layer's
  hidden state for the same prompt; the worst recorded layer difference is
  `5.53e-5`. The logit test enforces the greedy token sequence and a loose `5.0`
  maximum absolute logit guard; `4.58e-5` is the observed first-step difference,
  not the enforced threshold.
- Golden CPU token tests also cover `1+1=` and `Once upon a time`.

## Phase 3 prefill

### Environment

Same model, device, ROCm/HIP, and host as the top of this document. Conditions
for the prefill measurements: Qwen3.5-4B Q4_K_M, `gfx1151`, greedy sampling,
`chunk-size` 512 except where a full prompt chunk is smaller, two warmup
iterations for the GEMM sweep and one for prompt curves unless stated
otherwise. The prompt text is the repeated-prose corpus used by `bench`.

### Prefill GEMM path comparison

`build/yuuka_prefill_sweep` compares, for one GEMM submission followed by a
completion wait:

- `row`: one plan containing `M` single-row GEMV operations (the pre-Phase-3
  prefill path);
- `batch`: the Phase 2 batched GEMV (`M <= 16`);
- `wmma`: the custom rocWMMA kernel with FP32 activations converted to FP16 in
  LDS;
- `rocblas`: the rocBLAS FP16 baseline, including the FP32-to-FP16 cast
  submitted immediately before the GEMM.

Values are milliseconds per submission, the mean of five measured iterations
after two warmups. `N` is output width and `K` is input width. Host timing
includes the completion wait.

`Q4_K`, milliseconds:

| M | 2560→9216 row | batch | wmma | 9216→2560 row | batch | wmma |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.364 | 0.364 | 0.362 | 0.315 | 0.333 | 0.685 |
| 4 | 1.385 | 0.572 | 0.376 | 1.231 | 0.507 | 0.730 |
| 8 | 2.748 | 0.862 | 0.384 | 2.447 | 0.795 | 0.711 |
| 16 | 5.468 | 1.461 | 0.400 | 4.876 | 1.298 | 0.724 |
| 32 | 10.945 | — | 0.392 | 9.752 | — | 0.813 |
| 64 | 21.902 | — | 0.734 | 19.448 | — | 0.900 |
| 128 | 43.823 | — | 1.520 | 38.978 | — | 1.261 |

`Q6_K`, milliseconds:

| M | 2560→9216 row | batch | wmma | 9216→2560 row | batch | wmma |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.283 | 0.359 | 0.418 | 0.261 | 0.328 | 0.688 |
| 4 | 1.106 | 0.593 | 0.429 | 1.032 | 0.528 | 0.726 |
| 8 | 2.294 | 0.888 | 0.432 | 2.103 | 0.787 | 0.698 |
| 16 | 4.618 | 1.471 | 0.443 | 4.279 | 1.322 | 0.739 |
| 32 | 9.253 | — | 0.474 | 8.574 | — | 0.845 |
| 64 | 19.526 | — | 0.974 | 17.488 | — | 1.058 |
| 128 | 38.525 | — | 1.861 | 35.854 | — | 1.451 |

`F16`, milliseconds, including the cast for `rocblas`:

| M | 2560→9216 row | batch | wmma | blas | 9216→2560 row | batch | wmma | blas |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.295 | 0.370 | 0.606 | 0.313 | 0.286 | 0.389 | 1.149 | 0.290 |
| 4 | 1.163 | 0.577 | 0.695 | 0.930 | 1.125 | 0.839 | 0.987 | 0.573 |
| 8 | 2.310 | 0.919 | 0.705 | 0.942 | 2.261 | 1.401 | 1.640 | 0.575 |
| 16 | 4.571 | 1.606 | 0.716 | 0.977 | 4.498 | 2.162 | 1.138 | 0.582 |
| 32 | 9.158 | — | 0.831 | 1.042 | 9.052 | — | 1.500 | 0.602 |
| 64 | 18.245 | — | 1.537 | 1.476 | 18.128 | — | 1.301 | 1.079 |
| 128 | 36.424 | — | 2.751 | 1.612 | 35.975 | — | 1.581 | 2.298 |

`Q8_0`, `Q5_K`, and `Q4_K` follow the same crossover: the small-row paths are
fastest at `M = 1` (per-row GEMV for `Q8_0`, batched GEMV for `Q5_K`, and
effectively tied for `Q4_K`); WMMA wins from `M = 4` for wide projections
(`K <= N`) and from `M = 8` for tall projections (`K > N`). Rows 2560→9216 and
8192→2560 were measured; the shorter `Q8_0` table is omitted because the
selected path is identical. `SelectPrefillPath` uses batched GEMV below four
rows for every dtype; at `M = 1` the sweep shows per-row submission is faster
for `F16`, `Q8_0`, and `Q6_K` (about 20-30%), so the small-row dispatch remains
a follow-up.

The dispatch table in `backend/hip.SelectPrefillPath` encodes these crossovers.
For F16 it selects batched GEMV below `M = 8` for wide projections, rocBLAS
where the measured baseline wins, and WMMA otherwise; for quantized dtypes it
selects batched GEMV only at the measured small-row points. Differences below
5% are treated as run-to-run noise: the table selects rocBLAS only where it was
decisively faster and WMMA otherwise. The 2560→9216 `M = 64` case is a 4%
rocBLAS advantage left on WMMA by that rule; the tall `M = 4` cases select
batched GEMV because it is decisively faster than WMMA.

### Prompt-length performance

`yuuka` HIP, Qwen3.5-4B Q4_K_M, greedy sampling, chunk size 512. Prefill and
decode use the same process; context capacity is `prompt + 16 + 64`. Prompts up
to 2048 use one warmup and three measured repetitions; 8192 and 32768 use one
measured repetition without warmup because each run is long. P50 values:

| Prompt tokens | Prefill | Prefill tok/s | TTFT | Decode tok/s (16 tokens) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.409 s | 313.05 | 0.409 s | 9.70 |
| 512 | 1.635 s | 313.06 | 1.636 s | 9.16 |
| 2048 | 6.690 s | 306.14 | 6.690 s | 7.47 |
| 8192 | 37.687 s | 217.37 | 37.687 s | 4.31 |
| 32768 | 340.357 s | 96.28 | 340.357 s | 1.60 |

Prefill throughput is flat through 2048 tokens and then declines as attention
work grows with context length. Decode throughput declines for the expected
reason: the Phase 2 decode-attention kernel is linear in the cached context and
is not part of the Phase 3 prefill work. The 32768 run completed with correct
state; its long duration is a measured limitation of the scalar Flash Attention
tile, not a failure or truncation.

### llama.cpp comparison

Same GGUF (`00fe7986...`), same device, greedy prompt processing only
(`llama-bench -n 0`). `llama.cpp` used its default prompt batch 2048 and ubatch
512 with flash attention `auto`; `yuuka` used chunk size 512. The build is
`strix-llama.cpp` commit `c8912d7f5` (build 12216, version 0.6.0-dev). HIP used
ROCm/HIP 7.16 on gfx1151; Vulkan used Mesa RADV 25.0.7 on gfx1151. Each
`llama-bench` cell is the mean and standard deviation of two measured
repetitions after its default warmup.

| Prompt tokens | yuuka HIP tok/s | llama.cpp HIP tok/s | HIP ratio | llama.cpp Vulkan tok/s | Vulkan ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 313.05 | 1589.53 ± 265.12 | 5.1x | 2154.59 ± 5.92 | 6.9x |
| 512 | 313.06 | 2163.12 ± 94.78 | 6.9x | 2307.82 ± 7.93 | 7.4x |
| 2048 | 306.14 | 2171.11 ± 24.63 | 7.1x | 2235.83 ± 16.82 | 7.3x |
| 8192 | 217.37 | 2077.64 ± 10.40 | 9.6x | 2114.75 ± 22.70 | 9.7x |
| 32768 | 96.28 | 1494.85 ± 12.68 | 15.5x | 1702.18 ± 31.82 | 17.7x |

The custom path does not close the gap to llama.cpp. The gap is widest at long
context, where the scalar Flash Attention tile and its single-wave-per-query
structure dominate. At short and mid context the remaining gap is smaller. The
GEMM itself is weight-reuse limited by its 32-row M tile: a 512-row chunk
reloads each weight tile 16 times. llama.cpp's HIP and Vulkan kernels use
larger tiles, matrix-core layouts, and pipelined loads. Closing the GEMM gap
requires a larger M tile (with more accumulator fragments per wave) and
double-buffered LDS; closing the long-context gap additionally requires a
matrix-core or multi-query-tile attention kernel. These are recorded as
follow-up work, not claimed here.

The comparison uses different prompt tokenization only in the sense that each
runner builds its own prompt from the same token count; both use the same GGUF
tokenizer and greedy prompt processing. `llama-bench` reports prefill only, so
these ratios are not end-to-end generation comparisons.

### Short-prompt latency and correctness

The Phase 2 baseline at the top of this document submitted 16 prompt tokens
through the decode graph and measured 1.433 s prefill (11.17 tok/s) and 9.87
decode tok/s on HIP. With chunked prefill on the same model and prompt length,
`bench --prompt-tokens 16 --decode-tokens 8 --chunk-size 16` measures 0.137 s
prefill (117.02 tok/s) and 9.88 decode tok/s. Decode is unchanged; short-prompt
TTFT improved by 10.5x.

Chunked prefill correctness is checked at two levels:

- `tests/hip/prefill_test.cpp` compares the fused chunk kernels, WMMA GEMM, and
  Flash Attention against independent float references.
- `engine/gpu/prefill_test.go` runs the tiny hybrid fixture with chunk sizes 1,
  3, 7, 16, 17, 33, and 64, requires the same top-1 token as the CPU reference,
  and enforces a `2e-3` maximum logit difference. It also decodes one token
  after each chunked prefill and compares that logit vector with the CPU state
  produced by a single unchunked prefill.
- `TestGPUChunkedPrefillLongContext` prefills 1024 real-model tokens with chunk
  sizes 128, 512, and 1024. The three logit vectors are bitwise identical
  (maximum absolute difference `0`) and select the same top-1 token. This is
  the KV/recurrent-state equivalence check at a context length where the
  earlier shared-scratch race was observable; the fixed kernel is also checked
  for bitwise repeatability in the native test.

A 32-token real-model prompt ("Write a short list of colors...") produces the
same greedy continuation with chunk sizes 7, 16, and 32 as the CPU reference.
Chunk 7 exercises a partial final chunk (7,7,7,7,4) and mixes WMMA for wide
projections with batched GEMV for tall ones; chunks 16 and 32 select WMMA for
every projection. Prefill time is 0.526 s, 0.265 s, and 0.167 s respectively.

### Kernel resource usage

`scripts/kernel_resources.sh` compiles the prefill kernels with
`-Rpass-analysis=kernel-resource-usage`. All listed kernels report zero scratch,
zero VGPR spills, and 16 waves/SIMD occupancy.

| Kernel | VGPR | SGPR | Static LDS | Dynamic LDS |
| --- | ---: | ---: | ---: | ---: |
| `gemm_wmma_kernel<F32>` | 43 | 26 | 14,336 B | 0 |
| `gemm_wmma_kernel<F16>` | 43 | 26 | 14,336 B | 0 |
| `gemm_wmma_kernel<Q8_0>` | 45 | 26 | 14,336 B | 0 |
| `gemm_wmma_kernel<Q4_K>` | 47 | 33 | 14,336 B | 0 |
| `gemm_wmma_kernel<Q5_K>` | 47 | 34 | 14,336 B | 0 |
| `gemm_wmma_kernel<Q6_K>` | 39 | 34 | 14,336 B | 0 |
| `attention_prefill_f32_kernel` | 53 | 40 | 0 | 32,768 B at head_dim 256 |
| `gdn_chunk_f32_kernel` | 37 | 70 | 8 B | 2,048 B at state 128 |
| `rmsnorm_batch_f32_kernel` | 9 | 18 | 1,024 B | 0 |
| `rmsnorm_heads_batch_f32_kernel` | 9 | 20 | 1,024 B | 0 |
| `rope_batch_f32_kernel` | 19 | 18 | 0 | 0 |
| `kv_write_batch_f32_kernel` | 14 | 34 | 0 | 0 |
| `embedding_batch_f32_kernel` | 20 | 19 | 0 | 0 |
| `conv1d_chunk_f32_kernel` | 26 | 34 | 0 | 0 |
| `cast_f32_to_f16_kernel` | 7 | 18 | 0 | 0 |

Dynamic LDS is set by the launcher and does not appear in the compiler's static
figure.

### Phase 3 reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka

build/yuuka_prefill_sweep
scripts/kernel_resources.sh

for p in 128 512 2048 8192 32768; do
  /tmp/yuuka-hip bench \
    --model /path/to/Qwen3.5-4B-Q4_K_M.gguf --backend hip \
    --prompt-tokens "$p" --decode-tokens 16 \
    --warmup 1 --reps 3 --chunk-size 512
done

# llama.cpp comparison, same GGUF and device.
build/bin/llama-bench -m /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  -p 128,512,2048,8192,32768 -n 0 -r 2 -o md
LD_LIBRARY_PATH=$PWD/build-vk/bin build-vk/bin/llama-bench \
  -m /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  -p 128,512,2048,8192,32768 -n 0 -r 2 -o md
```

The llama.cpp Vulkan build requires the Vulkan loader and a gfx1151 ICD. The
recorded run used Mesa RADV 25.0.7. Warmup and repetition counts differ for the
longest prompts in the yuuka curves; the values reported below state the actual
count used.


## Phase 4 serving

### Environment and protocol

| Field | Value |
| --- | --- |
| Date | 2026-11-11 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf` |
| Model SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Model size | 2,740,937,888 bytes |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| Backend | HIP, paged FP32 KV, block size 16 |
| Context | 256 positions |
| Sampling | Greedy, temperature 0 |
| Prompt | 64 tokens from repeated English prose |
| Decode | 16 tokens per request |
| Prefix cache | disabled unless stated |
| Graph replay | enabled unless stated |
| Warmup | one request for latency/throughput runs; none for prefix runs |

Measurements use `bench/serving`, which reports queue wait, TTFT, prefill,
decode, and sampling separately. Run one benchmark process at a time; two
processes sharing the GPU invalidate both results.

### Single-user latency and graph replay

Concurrency 1, 3 measured requests, one warmup request.

| Configuration | Latency P50 | TTFT P50 | Prefill P50 | Decode P50 | Sampling P50 | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph replay | 1.793 s | 0.249 s | 0.241 s | 1.533 s | 1.61 ms | 8.93 |
| Plain stream | 1.796 s | 0.258 s | 0.249 s | 1.530 s | 1.68 ms | 8.90 |

Graph replay changes decode time by less than run-to-run noise at this context,
so the server default is the plain stream path. Queue wait was about 2 µs in
both runs.

### Multi-request throughput

Concurrency 4, 8 measured requests, 64 KV blocks, graph replay enabled.

| Metric | P50 | P95 |
| --- | ---: | ---: |
| End-to-end latency | 6.820 s | 7.324 s |
| Queue wait | 0.258 s | 0.351 s |
| TTFT | 0.557 s | 0.824 s |
| Prefill | 0.244 s | 0.248 s |
| Decode | 1.532 s | 1.534 s |
| Sampling | 1.66 ms | 1.71 ms |

Aggregate throughput was 0.557 requests/s and 8.91 output tokens/s over a
14.371 s wall interval. Per-request decode time is unchanged from the
single-request run. The scheduler interleaves request plans on one stream, so
concurrency improves admission fairness and TTFT under load but does not raise
aggregate token throughput; mixed-shape batched kernels are the missing
optimization.

### Prefix cache

Three identical 64-token prompts at concurrency 1 with the prefix cache enabled
and no warmup: one miss and two full-prompt hits. The first request's TTFT was
0.255 s; the two hits had TTFT P50 of 0.247 ms and zero prefill time because the
cached recurrent state and prompt logits were restored. Decode time remained
1.532 s. Aggregate throughput was 9.83 output tokens/s.

### Phase 4 correctness evidence

- `go test -tags hip ./engine/gpu ./model/qwen35 ./serving` with
  `YUUKA_TEST_MODEL` set passes on the Q4_K_M model with the paged kernels and
  graph replay enabled in the GPU serving tests.
- Concurrent CPU and HIP requests reproduce isolated-run token sequences and
  text at concurrency 2.
- Full and partial prefix hits reproduce a fresh run's tokens and text on both
  backends.
- Cancellation leaves zero used KV blocks and balanced allocator counters.
- Twelve sequential request cycles leave zero used blocks and no active
  requests.
- Native `hip_paged_graph_correctness` covers paged KV writes, paged decode and
  prefill attention against a float reference, device copies, dynamic graph
  replay at several positions, and rejection of non-dynamic plans.

### Phase 4 reproduction

```bash
go build -tags hip -o /tmp/yuuka-serving-bench ./bench/serving

# Single user, graph replay.
/tmp/yuuka-serving-bench --model /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --backend hip --ctx 256 --block-size 16 --kv-blocks 16 \
  --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph

# Same run on the plain stream path.
/tmp/yuuka-serving-bench --model /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --backend hip --ctx 256 --block-size 16 --kv-blocks 16 \
  --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph=false

# Four concurrent requests.
/tmp/yuuka-serving-bench --model /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --backend hip --ctx 256 --block-size 16 --kv-blocks 64 \
  --concurrency 4 --requests 8 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph

# Prefix reuse.
/tmp/yuuka-serving-bench --model /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --backend hip --ctx 256 --block-size 16 --kv-blocks 16 \
  --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 0 --graph --prefix-cache
```

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
