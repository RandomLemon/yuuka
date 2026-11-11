# GPU Kernels

## 1. Scope

This document specifies the device-side execution strategy for gfx1151:
operator families, prefill/decode separation, quantized GEMV and GEMM,
attention, fusion, dispatch, and numerical requirements. It does not specify
the Go scheduler or API; those are in [serving.md](serving.md).

Implementation status: decode-oriented kernels for `F16`, `Q8_0`, `Q4_K`,
`Q5_K`, and `Q6_K` GEMV, RMSNorm, L2Norm, gated RMSNorm, RoPE, decode attention,
convolution state, gated delta-net recurrence, embedding, SwiGLU, residual add,
and copy are implemented and validated against CPU references. A correctness
baseline `F32`/`F16` batched GEMM exists for prefill experiments, but the
quantized WMMA GEMM, Flash Attention, chunked prefill, and tuning tables are not
yet implemented.

The guiding rule is that **prefill and decode are separate operator families**.
They may share data layouts and helper primitives, but they are not the same
GEMM with a different `M`.

## 2. gfx1151 execution model

| Constraint | Kernel consequence |
| --- | --- |
| 40 CUs (reference configuration) | Tiles and workgroups must provide enough parallelism to fill the GPU; tiny grids waste the device. |
| Wave32-first | Prefer 32-lane workgroups and reductions tuned for Wave32. |
| RDNA WMMA | Use WMMA for prefill matrix tiles. Do not reuse CDNA MFMA kernels. |
| LDS capacity | Double-buffering and tile size trade against occupancy; avoid LDS spills. |
| VGPR budget | Aggressive fusion can reduce occupancy; measure VGPR usage per kernel. |
| UMA / LPDDR5x | Decode weights are read from shared system memory; effective bandwidth is the primary limit. |
| Kernel launch cost | Decode launches many small kernels; batching and Graph replay matter. |

Every kernel must be evaluated on the target device with recorded ROCm and
driver versions. Occupancy and bandwidth estimates from other RDNA parts are not
accepted as evidence.

## 3. Prefill versus decode

### 3.1 Prefill

Prefill processes the prompt with a relatively large `M`:

```text
Y[M, N] = X[M, K] * W[K, N]
```

The critical resources are matrix throughput, weight reuse across the `M`
dimension, attention I/O, and tile scheduling. The primary tools are WMMA,
quantized GEMM, Flash Attention, larger tiles, and software pipelining.

### 3.2 Decode

Decode generates one token at a time, or a small batch:

```text
y[N] = x[K] * W[K, N]       (per request, M = 1)
```

The critical resources are quantized weight-read bandwidth, KV-cache reads,
reduction efficiency, and launch overhead. The primary tools are fused
dequantize-GEMV, GQA-aware paged attention, and shape-dependent batching.

### 3.3 Roofline intuition

For a 4.5 GB weight set and a theoretical 256 GB/s memory bandwidth, a fully
streaming single-request decode has a rough upper bound near 57 tokens/s:

```text
256 GB/s / 4.5 GB ≈ 57 token/s
```

This is an order-of-magnitude estimate, not a benchmark. Real throughput is
lower and depends on effective bandwidth, KV reads, dequantization cost, kernel
efficiency, context length, and host scheduling. With larger batches, weight
reuse can make **aggregate throughput** exceed the single-request estimate.

### 3.4 Routing

| Condition | Preferred path |
| --- | --- |
| `M = 1` | Quantized GEMV, fused unpack/scale/reduce |
| Small batch (`M = 2..16`) | GEMV or small-M GEMM, chosen by measured threshold |
| Larger `M` | Quantized WMMA GEMM or rocBLAS/rocWMMA baseline |
| Prefill | WMMA GEMM + Flash Attention |
| Decode | GEMV + Paged Attention |

The thresholds are **measured per quantization type and shape class**, not
hard-coded from another GPU.

## 4. Operator family map

| Kernel family | Main scenario | Optimization focus |
| --- | --- | --- |
| `gemv_q4_k` | Single-token decode | Packed reads, fused dequantization, Wave reduction |
| `gemv_q6_k` | Decode | High-precision K-quant layout |
| `gemv_q8_0` | Decode / baseline | Simple, stable high-bandwidth quantization |
| `gemm_fp16_wmma` | Prefill baseline | Tiling, WMMA, LDS, pipelining |
| `gemm_q4_k_wmma` | Quantized prefill | Tiled dequantization plus WMMA |
| `gemm_q6_k_wmma` | Quantized prefill | Scale handling, tile reuse |
| `flash_attention` | Prefill | Online softmax, causal mask, GQA |
| `paged_attention` | Decode | Block table, shared KV heads, long-context split |
| `rmsnorm` / `rope` | All stages | Minimize memory traffic and launches |
| `fused_swiglu` | FFN | SiLU, gate, and multiply |
| `sampling` | Output | GPU sampling or a low-cost host path |

## 5. Decode: quantized GEMV

### 5.1 Data flow

```text
GGUF packed Q4_K / Q6_K / Q8_0
            |
 Coalesced read + unpack + scale
            |
 Multiply-accumulate in registers
            |
 Wave / block reduction
            |
 Output FP16 / FP32
```

The kernel must not expand the full weight matrix into FP16 before computation.
Unpack, scale application, multiply-accumulate, and reduction are fused as far
as the register and LDS budget allows.

### 5.2 Q4_K specifics

Q4_K uses 256-weight superblocks with encoded scales and mins. Implementation
requirements:

- respect superblock and sub-block boundaries;
- load packed words with alignment-safe, coalesced accesses;
- apply the correct scale/min encoding per sub-block;
- map workgroups across output channels and the reduction dimension so that
  neighboring lanes read neighboring data;
- keep partial sums in registers and reduce within a wave where possible;
- avoid per-element branches on quantization metadata.

### 5.3 Workgroup mapping

The main design axes to benchmark:

- one wave per output channel versus multiple output channels per workgroup;
- vectorized loads versus narrower packed loads;
- register reuse versus LDS staging for scales and partial sums;
- batch sizes 1, 2, 4, 8, and 16 to find the GEMV/GEMM crossover;
- accumulation precision and deterministic reduction order.

### 5.4 Precision

For FP16/BF16 inputs, accumulate in FP32 unless a measured, documented
tradeoff justifies otherwise. Quantization scales and mins must be applied
before the final rounding step; avoid double rounding between unpack and
accumulate. The output dtype is part of the execution plan, not an implicit
kernel default.

## 6. Prefill: WMMA and quantized GEMM

### 6.1 Baseline

The initial prefill path uses rocBLAS or rocWMMA FP16/BF16 matrix multiplication
as the performance and correctness baseline. This provides a stable reference
while custom kernels are developed.

### 6.2 Quantized GEMM

A custom quantized WMMA kernel follows this structure:

1. Load a tile of `X` into LDS.
2. Stream the corresponding GGUF-quantized tile of `W` and unpack it.
3. Convert the unpacked values into the register or LDS layout expected by WMMA.
4. Execute WMMA operations and accumulate.
5. Write only the final output tile; never materialize a full dequantized weight
   matrix.

The kernel must handle the quantization metadata layout without scattering
scales across workgroups inefficiently. Tile shape, K-step, and LDS buffering
are chosen per quantization type.

### 6.3 Dispatch

Dispatch is based on `M`, `N`, `K`, quantization type, and available scratch.
A custom quantized GEMM is not assumed to beat the library path for every shape.
The planner uses an offline, measured dispatch table and falls back to the
baseline when the custom kernel is unsupported or slower.

## 7. Attention

### 7.1 Prefill: Flash Attention

- Tile Q, K, and V; use online softmax.
- Support causal masking and GQA.
- Do not materialize the full attention matrix.
- Support chunked prefill so long prompts do not block decode indefinitely.
- Tune tile sizes by context length and head dimension.

### 7.2 Decode: Paged Attention

- Locate KV blocks through the per-request block table.
- Exploit shared KV heads in GQA.
- Use fewer workgroups for short contexts; use split-KV plus a reduction pass
  for long contexts.
- Avoid low occupancy and excessive random reads.
- Reserve metadata for sliding-window and hybrid-attention models.

| Aspect | Prefill | Decode |
| --- | --- | --- |
| Query length | Prompt chunk | 1 token (or small batch) |
| Main limit | Matrix throughput and attention I/O | KV reads, weight bandwidth, launch overhead |
| KV access | Contiguous prompt range | Block table indirection |
| Parallelism | Tiles over query/key dimensions | Heads and KV splits |
| Masking | Causal, chunk-aware | Causal with cached prefix |

The logical KV view is block-based and independent of physical placement.
Kernel-specific block layouts are backend-internal.

## 8. Fusion

Implement fusion in order of benefit and correctness risk:

1. Dequantization + GEMV/GEMM.
2. RMSNorm and necessary elementwise operations.
3. RoPE and Q/K writes.
4. SwiGLU activation and multiply.
5. Decode attention softmax + value reduction.

Every fusion must be evaluated on all of:

- kernel launch count;
- VGPR usage and occupancy;
- LDS consumption;
- memory traffic;
- end-to-end latency, not just isolated kernel time.

Over-fusion can reduce occupancy or increase register pressure enough to lose
performance. A fused kernel that wins on one shape may lose on another; retain
the unfused path as a fallback when measurements justify it.

## 9. Dispatch and tuning

### 9.1 Shape signature

Each kernel is registered with a signature that includes at least:

```text
operator, M, N, K, input dtype, weight dtype, batch, sequence class, backend
```

The planner resolves the signature to a kernel configuration. Signature lookup
must be cheap enough to perform during planning, not per token.

### 9.2 Kernel registry

The registry records:

- supported dtypes and quantization types;
- shape constraints (alignment, minimum dimensions, block divisibility);
- required scratch and workspace;
- graph-capture compatibility;
- the selected implementation (custom kernel or library fallback).

### 9.3 Offline tuning

The primary source of dispatch decisions is an offline tuning table measured on
gfx1151. Each table entry records the ROCm version, kernel version, shape class,
and measured result. Runtime autotuning is optional and must be bounded so it
does not dominate startup latency.

### 9.4 Fallback

If a custom kernel is unavailable, unsupported for a shape, or slower than the
library baseline, the planner selects the documented fallback. Silent numerical
substitution is forbidden. A fallback must be visible in diagnostics and
benchmarks.

## 10. Numerical requirements

- Define a tolerance per stage: tensor-level outputs, logits, and final token
  selection.
- Use stable softmax (max subtraction) and stable normalization.
- Handle non-aligned dimensions, zero-length edge cases, and extreme values.
- Check for NaN/Inf propagation and overflow in reduction accumulators.
- Specify whether a kernel is deterministic. Split reductions and atomics may
  produce run-to-run variation; tests must use a deterministic mode or an
  explicit tolerance.
- Compare against the CPU reference and at least one independent implementation
  for critical operators.
- Do not tune a kernel against a single prompt or a single shape.

## 11. Kernel acceptance criteria

A kernel is not complete until:

- [ ] It matches the CPU reference on random shapes, non-aligned sizes, and
      boundary cases.
- [ ] Its numerical tolerance is documented and tested.
- [ ] It has a benchmark across the relevant shape classes.
- [ ] Its VGPR, LDS, and occupancy characteristics are recorded.
- [ ] The planner can select it and fall back from it.
- [ ] It is integrated into an end-to-end model run.
- [ ] It does not regress a previously accepted shape class.
