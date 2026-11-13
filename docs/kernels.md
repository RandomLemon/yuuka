# GPU Kernels

## 1. Scope

This document specifies the device-side execution strategy for gfx1151:
operator families, prefill/decode separation, quantized GEMV and GEMM,
attention, fusion, dispatch, and numerical requirements. It does not specify
the Go scheduler or API; those are in [serving.md](serving.md).

Implementation status: decode-oriented kernels for `F16`, `Q8_0`, `Q4_0`,
`Q5_0`, `Q4_K`, `Q5_K`, and `Q6_K` GEMV, RMSNorm, L2Norm, gated RMSNorm, RoPE,
decode attention, convolution state, gated delta-net recurrence, embedding,
SwiGLU, residual add, and copy are implemented and validated against CPU
references. Batched GEMV
reuses each packed weight element across up to 16 rows; its measured thresholds
are encoded in `backend/hip.SelectGemvPath`.

Phase 3 adds the prefill family: rocWMMA FP16 and quantized GEMM for `F16`,
`F32`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K`; a rocBLAS FP16 baseline with an
explicit FP32-to-FP16 cast; causal GQA Flash Attention over prompt chunks;
chunked prefill with per-chunk KV writes; row-batched RMSNorm, head RMSNorm,
RoPE, embedding, and KV-write kernels; and a fused gated delta-net chunk kernel
that performs convolution, L2 normalization, state update, and gated RMSNorm in
two launches per layer. `backend/hip.SelectPrefillPath` is the measured dispatch
table described in section 9.

Phase 4 adds the paged KV family: decode and prefill attention kernels that
resolve every key/value position through an int32 block table, paged KV-write
kernels for one row and for a chunk, dynamic-parameter variants that read the
token and position from device memory, and a C ABI (version 4) for
device-to-device block copies and HIP Graph capture/replay. KV pools support
F32 (default), F16, and Q8_0; see section 7.3. Attention remains scalar per lane
rather than WMMA-based, and continuous batching is scheduler-level interleaving
of per-request plans; mixed-shape prefill/decode kernel batches and split-KV
decode are not implemented.

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
| `gemv_q4_0` / `gemv_q5_0` | Decode / broader GGUF coverage | Nibble and high-bit unpack, batch reuse |
| `gemm_fp16_wmma` | Prefill baseline | Tiling, WMMA, LDS, pipelining |
| `gemm_q4_k_wmma` | Quantized prefill | Tiled dequantization plus WMMA |
| `gemm_q6_k_wmma` | Quantized prefill | Scale handling, tile reuse |
| `flash_attention` | Prefill | Online softmax, causal mask, GQA |
| `paged_attention` | Decode | Block table, shared KV heads, long-context split |
| `rmsnorm` / `rope` | All stages | Minimize memory traffic and launches |
| `fused_swiglu` | FFN | SiLU, gate, and multiply |
| `moe_route` / `moe_expert_*` | Sparse MoE | Top-k routing, per-expert assignment bins, packed expert reads |
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

### 5.4 Q4_0 and Q5_0

Q4_0 and Q5_0 use 32-value blocks. Q4_0 stores one FP16 scale and sixteen packed
nibble bytes; values are `d * (q - 8)`. Q5_0 adds a four-byte high-bit plane;
values are `d * (q - 16)`. Both are supported by the CPU reference, single and
batched HIP GEMV, embedding, and WMMA prefill. Q4_1/Q5_1 and the IQ families are
still capability failures.

### 5.5 Precision

For FP16/BF16 inputs, accumulate in FP32 unless a measured, documented
tradeoff justifies otherwise. Quantization scales and mins must be applied
before the final rounding step; avoid double rounding between unpack and
accumulate. The output dtype is part of the execution plan, not an implicit
kernel default.

## 6. Prefill: WMMA and quantized GEMM

### 6.1 Implementations

Prefill uses a separate operator family from decode. The runtime selects one of:

- **Batched GEMV**: the Phase 2 kernel, limited to 16 rows. It is retained as
  the baseline for very small prompt chunks.
- **rocWMMA GEMM**: the custom kernel in `kernels/hip/gemm/wmma_gemm.hip`. It
  accepts FP32 activations and every supported weight dtype.
- **rocBLAS FP16**: the external baseline in `kernels/hip/gemm/gemm_blas.hip`.
  It requires an FP16 activation buffer; the planner inserts an explicit cast.

All three are validated against a float reference built from the exact packed
weights. The rocBLAS baseline is a comparison point and a dispatch choice for
the measured F16 shape classes, not a fallback for unsupported quantization.

### 6.2 Custom WMMA GEMM

`gemm_wmma_kernel` computes a 32x64 output tile with eight waves. Each wave owns
one 16x16 fragment and accumulates in FP32 across the K dimension.

1. Load a 32x32 FP32 `X` tile and convert it to FP16 in shared memory.
2. Stream the corresponding 64x32 weight tile and decode one 32-value
   quantization sub-block per wave lane directly into the FP16 shared tile.
3. Run two 16x16x16 WMMA operations per wave per K step.
4. Stage the FP32 accumulator tile in shared memory and write only the valid
   output elements, so non-aligned `M` and `N` edges are safe.

The K step is 32 because every K-quant superblock splits into 32-value
sub-blocks. A segment therefore uses exactly one scale/min pair (or two 16-value
Q6_K scales) and never crosses a superblock. No dequantized weight matrix is
materialized. `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `F16`, and `F32` share the same
tile path; only the packed-load helper differs.

Static shared memory is 14 KiB per workgroup: 2 KiB A tile, 4 KiB B tile, and
8 KiB output staging. Compiler-reported usage is 39-47 VGPRs, 26-34 SGPRs, no
spills, and a 16 waves/SIMD resource maximum. The kernel is not pipelined; LDS
double-buffering and a larger M tile for weight reuse across workgroups are
later optimizations.

### 6.3 Dispatch

Dispatch is based on `M`, `N`, `K`, quantization type, and the measured row
class. The table is in `backend/hip.SelectPrefillPath`; measurements are in
[benchmarks.md](benchmarks.md).

| Condition | Selected path |
| --- | --- |
| `M < 4` | Batched GEMV |
| Quantized, `M < 8`, and `K > N` | Batched GEMV |
| Quantized, other `M >= 4` | Custom rocWMMA |
| F16, `4 <= M <= 64`, `K > N` | rocBLAS + cast |
| F16, `M < 8`, `K <= N` | Batched GEMV |
| F16, `M >= 128`, `K <= N` | rocBLAS + cast |
| F16, other `M >= 8` | Custom rocWMMA |

The custom kernel is selected only where the sweep showed it faster than the
compared baseline. F16 baselines include the cast cost. A dispatch table entry
is valid only for the recorded gfx1151 environment; the kernel is not selected
from assumptions about other RDNA parts.

The isolated GEMM is weight-reuse limited: with a 32-row M tile, each
workgroup reloads the weight tile, so a 512-row chunk reads every weight tile
16 times. The end-to-end prompt path measured about 313 tok/s at 512 tokens
after the recurrent-kernel fix, versus llama.cpp HIP at about 2163 tok/s on the
same GGUF and device. That remaining gap is not fully attributed in this phase;
the 32-row M tile, scalar prefill attention, and launch structure are
candidates. A larger M tile with more accumulator fragments per wave and
double-buffered LDS is the next GEMM optimization; it is not implemented and is
not claimed in the benchmark results.

## 7. Attention

### 7.1 Prefill: Flash Attention

`attention_prefill_f32_kernel` processes a prompt chunk with one wave per query
row. A workgroup owns eight query rows for one head.

- Q is read into registers, with each lane holding `ceil(head_dim/32)` values.
- K is staged one 32-key block at a time in dynamic shared memory
  (`32 * head_dim * 4` bytes; 32 KiB for head_dim 256). V is read from global
  memory with coalesced per-lane vectors.
- Scores are reduced within the wave; online softmax keeps per-row max, sum,
  and FP32 value accumulators, so the full attention matrix is never
  materialized.
- Causality uses the chunk position: query row `r` may attend to keys
  `0..position+r`. Waves in a workgroup share the K-block loop to keep
  `__syncthreads` convergent and mask keys beyond their own position.
- GQA maps query head `h` to KV head `h / (heads/kv_heads)`.
- The Qwen3.5 attention gate (the second half of each query vector) is applied
  as `sigmoid(gate)` before the output write.

Supported prefill head dimensions are 1-256; the plane-stage kernel reports an
unsupported capability error above 256. Decode attention remains the
per-head kernel described in section 5. Flash Attention is not used for decode.

Chunked prefill splits a prompt into at most `chunk-size` rows (default 512,
maximum 4096). KV writes for a chunk happen before its attention, so later
chunks see the full causal prefix. The runtime builds and caches one plan per
chunk length; only the final chunk executes the output tail.

### 7.2 Decode: Paged Attention

`attention_paged_f32_kernel` runs one workgroup per query head:

- The workgroup stages the query head in shared memory and reduces the score
  dot product across 256 threads.
- Each logical position `p` maps to
  `block_table[p / block_size] * block_size + p % block_size`; K and V are read
  from the pool through that row. The block table and pool are device buffers.
- Scores for one head use a per-head stride so one scores workspace serves all
  heads. The causal length is `position + 1`.
- GQA maps query head `h` to KV head `h / (heads/kv_heads)`.
- The attention gate is applied as `sigmoid(gate)` before the output write.
- The dynamic variant reads `position` from a device parameter buffer so a
  captured HIP Graph can be replayed without changing kernel arguments.

`attention_prefill_paged_f32_kernel` keeps the same eight-query-row workgroup
tiling and online softmax as the contiguous prefill kernel, but resolves each
staged K row and each V read through the block table. `launch_kv_write_paged_f32`
and `launch_kv_write_paged_dynamic_f32` map written positions through the same
table; the batch case is selected by the row count, not a separate kernel name.

### 7.3 KV cache formats

The paged KV family accepts an F32, F16, or Q8_0 pool selected by the plan. The
same block table and pool geometry are used for every format; only the row byte
size changes:

| Format | Row layout |
| --- | --- |
| F32 | `kv_dim` FP32 values |
| F16 | `kv_dim` FP16 values |
| Q8_0 | `ceil(kv_dim/32)` groups of one FP16 scale plus 32 int8 values |

`kv_write_paged` quantizes per 32-value group for Q8_0; attention kernels decode
K and V through the same row format. Prefix-cache keys include the format so
cached blocks are never reused across formats.

### 7.4 Dense attention variant

The original paged attention kernels assume the Qwen3.5 query layout with a
sigmoid output gate in the second head-sized half. Qwen3 MoE uses a dense query
row without that gate. The plan carries a dense flag in `u[3]`; the dense kernel
skips the gate and uses `heads * head_dim` query stride. The Qwen3.5 path keeps
the gated variant.

Current limits:

- Q8_0 KV is supported but not faster than F32 at short context; its benefit is
  cache memory.
- One workgroup per head; split-KV and WMMA attention are not implemented.
- No mixed prefill/decode batch. Each step belongs to one request plan.
- Block-table entries are immutable for a request; all blocks are allocated at
  admission, so replay and kernels never observe a changing table.

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

### 8.1 MoE expert pipeline

`moe_route` selects top-k experts, normalizes the selected softmax weights, and
bins `(row, slot)` assignments into one bucket per expert using atomics. The
grouped kernels then index blocks by `(expert, output row)` and walk that
expert's assignment list:

- `moe_gate_up_grouped` reads one packed gate and up row per block and computes
  all assigned token rows, so the packed weight row is reused across rows.
- SwiGLU uses the existing fused `silu_mul` kernel.
- `moe_down_grouped` performs the expert down projection over the same buckets.
- `moe_reduce_add` applies routing weights and accumulates into the residual
  stream.

The routing and assignment buffers are sized `experts * rows * used`, which is
safe because one row contributes exactly `used` assignments. A malformed plan
that would exceed a bucket is rejected at plan construction. The pipeline is
validated with F32 and Q4_K synthetic fixtures; the Q4_K fixture covers chunked
prefill and prefill-plus-decode against the CPU reference.

### 8.2 Speculative verification

Speculative decoding reuses the chunked prefill graph with a tail that computes
logits for every candidate row instead of only the last row. Verification reads
those rows, accepts the matching greedy prefix, and on a mismatch restores the
session snapshot and replays the accepted prefix plus the correction token.
This is exact for greedy sampling and does not change the F32 KV or decode
kernels.

Implemented fusions, chosen because they remove launches or intermediate
traffic without reducing occupancy:

- Dequantization + GEMV and dequantization + WMMA GEMM.
- RMSNorm row batching, head batching, and fused RoPE over a chunk.
- KV-write batching for a prompt chunk.
- Prefill attention online softmax, value reduction, and attention gate.
- Gated delta-net chunk: convolution state update, q/k L2 normalization, the
  128x128 recurrent update, and gated RMSNorm per head. It replaces five
  launches and one normalized intermediate per token with two launches per
  layer. Each thread owns one 128x128 state column, and a barrier separates the
  q and k block reductions because both reuse the same shared reduction
  scratch. Both choices are required for deterministic multi-row execution;
  the row-major state layout is unchanged and shared with the decode kernel.

SwiGLU remains an elementwise pass over the chunk; fusing it into the GEMM
epilogue was not attempted because the measured prefill bottleneck is the
weight-streaming GEMM. Every fusion must be evaluated on all of:

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

Dispatch decisions and plan validation use these shape and capability fields:

```text
operator, M, N, K, input dtype, weight dtype, batch, sequence class, backend
```

There is no generic kernel registry in this release. Go selectors map the
fields they need to a concrete path, and plan construction validates supported
dtypes, row layout, buffer ranges, and dimensions before submission. Lookup is
per plan, not per tensor per token.

### 9.2 Dispatch selection and validation

Dispatch is recorded in two explicit Go selectors:

- `backend/hip.SelectGemvPath` chooses per-row or batched GEMV from dtype and
  batch size;
- `backend/hip.SelectPrefillPath` chooses batched GEMV, custom rocWMMA, or the
  rocBLAS FP16 baseline from dtype, rows, and the input/output widths.

Each plan addition validates the dtype, quantization row layout, shape
constraints, and buffer ranges. Unsupported combinations fail during planning;
there is no size-keyed registry, scratch registry, or graph-compatibility
registry to consult at runtime.

### 9.3 Offline tuning

Dispatch decisions are offline. Thresholds measured on gfx1151 are encoded in
`backend/hip.SelectGemvPath` and `backend/hip.SelectPrefillPath`, keyed by
operator family, dtype, shape class, and batch class, and recorded with the
measurement environment in [benchmarks.md](benchmarks.md). There is no runtime
autotuner. Adding a device requires re-running the sweeps and updating the
selectors and tests.

### 9.4 Fallback

If a custom kernel is unavailable, unsupported for a shape, or slower than the
library baseline, the planner selects the documented fallback. Silent numerical
substitution is forbidden. A fallback must be visible in diagnostics and
benchmarks.

## 10. Numerical requirements

- Prefill converts FP32 activations and dequantized weights to FP16 for WMMA
  input and accumulates products in FP32. This is a documented prefill policy,
  not an implicit fallback. Tensor tests compare against a float reference and
  use `2e-2 * (1 + |expected|)` for GEMM outputs; the observed worst case for
  the checked-in shapes is under `1.2e-2`.
- The prefill attention reference comparison uses
  `2e-4 * (1 + |expected|)`; the current checked-in shape reports about
  `3.0e-8` because both sides use FP32.
- Chunked prefill must produce the same greedy token as unchunked CPU prefill.
  The GPU tests compare final logits within `2e-3` and require an identical
  top-1 token for chunk sizes 1, 3, 7, 16, 17, 33, and 64 on the tiny fixture.
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

- [x] It matches the CPU reference on random shapes, non-aligned sizes, and
      boundary cases.
- [x] Its numerical tolerance is documented and tested.
- [x] It has a benchmark across the relevant shape classes.
- [x] Its VGPR, LDS, scratch, and occupancy figures are recorded for the Phase
      0-3 kernel families (`scripts/kernel_resources.sh`, results in
      [benchmarks.md](benchmarks.md)).
- [x] New Phase 5 kernels have recorded VGPR, LDS, and occupancy figures for the
      GEMV, WMMA, and MoE variants (`scripts/kernel_resources.sh`, results in
      [benchmarks.md](benchmarks.md)).
- [x] The planner can select it and fall back from it.
- [x] It is integrated into an end-to-end model run.
- [x] It does not regress a previously accepted shape class.
