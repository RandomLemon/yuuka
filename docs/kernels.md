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
are recorded in [benchmarks.md](benchmarks.md), and
`backend/hip.SelectPrefillPath` chooses the batched path for the small row
classes it covers.

Phase 3 adds the prefill family: rocWMMA FP16 and quantized GEMM for `F16`,
`F32`, `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K`; a rocBLAS FP16
baseline with an explicit FP32-to-FP16 cast; causal GQA Flash Attention over
prompt chunks; chunked prefill with per-chunk KV writes; row-batched RMSNorm,
head RMSNorm, RoPE, embedding, and KV-write kernels; and a gated delta-net
chunk operation that performs convolution, L2 normalization, state update,
and gated RMSNorm. `backend/hip.SelectPrefillPath` is the
measured dispatch table described in section 9.

Phase 4 adds the paged KV family: decode and prefill attention kernels that
resolve every key/value position through an int32 block table, paged KV-write
kernels for one row and for a chunk, dynamic-parameter variants that read the
token and position from device memory, and a versioned C ABI for
device-to-device block copies and HIP Graph capture/replay. KV pools support
F16 (default), F32, and Q8_0; see section 7.3. Prefill attention runs on the
FP16 MMA kernel for F32 and F16 pools whose shape it covers, and on the scalar
per-lane tiled kernel otherwise (section 7.5).
Continuous batching is scheduler-level interleaving of per-request plans;
mixed-shape prefill/decode kernel batches and split-KV decode are not
implemented.

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
| `gemv_row` / `gemv_row_pair` | Decode matvec (selected, one row, K % 512 == 0) | One wave per output row, sixteen K values per lane, two rows per wave, no shared staging and no barrier in K |
| `lm_head_stream` | Decode matvec fallback, both output projections | One wave per four output channels, 16-byte packed reads, LDS activation staging, wave reduction, no block barrier in K |
| `gemv_i8` | Decode matvec, opt-in and not selected | Same mapping over int8 digits, one `v_dot4` per four weights, per-sub-block scale and minimum |
| `lm_head_stream` epilogues | Decode matvec (selected) | Same streaming kernel with the residual add in the epilogue, or walking a second packed matrix to form the SwiGLU product (`GEMV_RESIDUAL`, `GEMV_PAIR`) |
| `recurrent_token` | Gated delta-net per-token state (selected) | Head state in registers across four thread groups, decay and update fused in one pass, gated RMSNorm folded in |
| `conv_l2norm` | Recurrent convolution and q/k L2 norm (selected) | One block per 256 channels, whole norm groups per block, ring state advanced in place |
| `gemv_q4_k` / `gemv_q6_k` / `gemv_q8_0` / `gemv_q4_0` / `gemv_q5_0` | Decode fallback, narrow projections, kernel reference | Packed reads, fused dequantization, block reduction |
| `gemm_fp16_wmma` | Prefill baseline | Tiling, WMMA, LDS, pipelining |
| `gemm_q4_k_wmma` | Quantized prefill | Tiled dequantization plus WMMA |
| `gemm_q6_k_wmma` | Quantized prefill | Scale handling, tile reuse |

| `flash_attention` | Prefill | Online softmax, causal mask, GQA |
| `attention_prefill_paged_blas_f16` | F16 prefill, 2048+ positions | Gathered K/V, rocBLAS QK/PV, FP32 softmax |
| `attention_paged_flash` / `attention_flash_combine` | Decode (selected) | Block table, one block per KV head and key tile, online softmax in registers, split partials merged by log-sum-exp |
| `attention_paged` (serial) | Decode fallback and reference | One block per query head, sequential positions, per-head score workspace |
| `rmsnorm` / `rope` | All stages | Minimize memory traffic and launches |
| `fused_swiglu` | FFN | SiLU, gate, and multiply |
| `moe_route` / `moe_gate_up_grouped` / `moe_down_grouped` | Sparse MoE prefill | Top-k routing, per-expert assignment bins, packed expert reads |
| `moe_gate_up_q8` / `moe_down_q8` | Sparse MoE prefill, IQ experts | Q8_1 activation tile, int8 `dp4a`, sixteen-assignment tile |
| `moe_gate_up_mmq` / `moe_down_mmq` | Sparse MoE prefill, K-quant and IQ experts | Q8_1 activation tile, int8 WMMA, 16-128 assignment tile, gather/scatter through the routed bucket |
| `moe_gate_up_selected` / `moe_down_selected` / `moe_shared_add` | Sparse MoE decode | One grid per routing slot, no binning, sigmoid-gated shared expert |
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

Two mappings exist and `backend/hip.SelectDecodeGemvPath` is the measured rule
between them. Both compute the same per-weight products through `decode_span4`;
they differ in how much of the machine one output row keeps busy.

**Per-row GEMV** (`gemv_*_kernel`): one 256-thread block per output row, one
packed element per thread per K step, a block-wide tree reduction, and for
Q4_K/Q5_K two barriers per 256-value step for the scale staging. It reads the
activation once per output row, so a Q4_K row reads 10 KiB of activation against
1.44 KiB of weights. Measured 25-76 GB/s on the decode shapes.

**Streaming** (`lm_head_kernel`, section 6.5): one wave owns four output
channels and walks the K axis; a lane reads 16 bytes of packed weights per step,
the activation row is staged in shared memory once per block, accumulators stay
in registers, and the only reduction is one wave shuffle per channel. No block
barrier sits inside the K loop. Measured 90-287 GB/s on the same shapes at 2.1-8.6
instructions per weight fewer, and 0.16 LDS instructions per weight against up to
2.66.

**Remapped row** (`gemv_row_kernel`, section 5.4): the one-row decode form of
the streaming family. One wave owns one output row, sixteen K values per lane
over a 512-value step, no shared-memory staging and no barrier inside K.

The rule, in output width at `n_in = 2560` and one row: the streaming family
wins from 512 outputs for Q4_K/Q5_K/Q6_K and from 1536 for Q8_0/F16 (5.3.1),
so the selector requires `n_out >= 512` for the K-quants and `n_out >= 1024`
for Q8_0/F16, a K step of 256 values, a dtype in the GEMV family, and 1-16 rows.
The K-quant threshold is the one their 512-output cell measured, which is where
the routed expert gate and up projections sit; Q8_0 and F16 keep the higher
threshold because their 1024-output cell is level with the per-row form and 1536
is their measured win. Within the family, a block of exactly one row whose K is
a multiple of 512 takes the remapped form and everything else takes the
channel-parallel form. Shapes outside the family keep the per-row kernel. The
32-output recurrent projections are the only decode shapes that land on the
fallback.

**Epilogues and the two-matrix form.** The same kernel carries two
plan-selected extensions. `GEMV_RESIDUAL` adds a residual row to every output
element in the epilogue, which folds the ADD that ends a sub-layer into the
projection and lets the output alias the residual row. `GEMV_PAIR` walks a second
packed matrix of the same dtype in the same block over the same staged activation
row, either writing both rows or forming `silu(A x) * (B x)`, which replaces
`ffn_gate`, `ffn_up`, and `SILU_MUL` with one operation. Both are plan-build-time
choices; the plain form stays for the int8 and per-row routes.

Three shapes of this idea were measured and rejected
([benchmarks.md](benchmarks.md#decode-fusion)): eight output
channels per wave (28% slower, the wider wave spills), a 512-value K step (7%
slower, the staging tile costs more than the barriers it removes), and a pair that
dispatches the decoder at run time so the two matrices could mix dtypes (40%
slower than the templated decode, because the switch dominates a loop that runs
0.3 instructions per weight). A pair of matrices whose widths differ by 8:1
(`q` and `k`) is also slower than two launches, because the grid follows the wider
matrix.

**Int8 variant, implemented and not selected.** `gemv_i8_kernel` and the
remapped `gemv_row_i8_kernel` keep the mapping but dot the four packed digits of
a lane against the Q8_1 activation words with `v_dot4`, applying the block scale
and the Q4_K/Q5_K minimum once per sub-block instead of once per weight. The
route is reachable through `Options.Int8Matvec` (fused) and
`Options.Int8Matvec` with `PrequantizedInt8` (one quantization operation per
activation group, which takes the remapped one-row kernel). On the remapped
mapping it measures 175.3 GB/s in step against 176.6 for the FP32 route and is
0.67 ms slower per decode step, because it loses the gate/up pair fusion and
pays for the activation quantizations. The FP32 route stays selected; the
numbers are in [benchmarks.md](benchmarks.md#int8-decode-matvec).

Numerics. The per-row kernel sums a strided set of K values per thread and
reduces over the block; the streaming kernel sums four neighbours per lane and
reduces across the wave. Both accumulate in FP32, so the difference is that
reassociation: `TestGPUStreamingDecodeMatchesGemv` measures 8.34e-06 maximum
absolute logit difference over eight greedy decode steps of the real model, with
identical tokens, and `tests/hip/gemv_test.cpp` checks every dtype against the
host dot product at 2e-3 relative and against the per-row kernel at 2.4e-4
absolute.

**Narrow pair.** The recurrent `beta` and `alpha` projections (Q8_0, 2560 -> 32)
are below the streaming crossover on their own, but they read the same
normalized row and share one `GEMV_PAIR` launch through the remapped pair
kernel, which halves the launch count of the pair. `decodeMatmulPair` allows that
one narrow case explicitly; every other below-threshold shape keeps the per-row
kernel.

### 5.4 Remapped one-row matvec

`gemv_row_kernel` in `kernels/hip/gemv/gemv_row.hip` is the selected decode
matvec. It changes the mapping rather than the arithmetic, so a regression is
attributable to dispatch rather than to numerics:

- Grid `ceil(n_out / 8)`: one block covers eight output rows (four waves, two
  rows per wave), so a 1024-output projection launches 128 blocks instead of 32.
- A lane owns sixteen consecutive K values and decodes them from one 8- to
  16-byte span of the packed row through `decode_span16`, which is the wide form
  of `decode_span4`: Q4_K and Q5_K spans are 16-byte aligned vector loads, F16
  and F32 spans are 32- and 64-byte reads, and Q6_K, Q8_0, Q4_0, and Q5_0 rows
  have strides that are not a multiple of sixteen bytes, so their spans are
  copied with four 32-bit loads.
- A wave steps K in 512-value steps, reads its sixteen activation values once
  per step, and reuses them for both of its rows. The activation row is small
  enough to stay in L1, so the kernel stages nothing in shared memory and has no
  barrier inside the K loop; the only sync is the wave shuffle tree per row.
- The epilogue adds the residual row when the plan fuses it, and
  `gemv_row_pair_kernel` walks a second packed matrix over the same activation
  values for the gate/up pair with the SwiGLU product folded in.

Measured in step at a 7-token context (`TestGPUProfileDecode`, profiled, 176.6
GB/s effective over the matvec family, against 130.3 GB/s before the remap):
`ffn_gate` 193.6, `lm_head` 229.8, `ffn_down` Q6_K 189.7, `attn_qkv` 178.8,
`ffn_down` Q4_K 158.4, `attn_q` 166.8, `attn_gate` 158.3, `ssm_out` 133.7,
`attn_output` 125.5, and `attn_k` 93.3 GB/s. The narrow and short-K shapes keep
a per-row cost that the wide shapes amortize: at 4096 input values a 2560-output
projection spends roughly 27 ns per output row on top of its weight bytes, which
is what leaves `attn_output` and `ssm_out` below the 190 GB/s target. The
channel-parallel form stays compiled for other row counts and K widths and as
the numerical reference.

#### 5.3.1 Measured decode shapes

One submission per plan, matrix resident, five warmups and twenty-five timed
submissions; `kernel` is the traced kernel time from
`scripts/decode_profile.sh`, which excludes the launch the host column includes.
These cells predate the remap in 5.4 and describe the channel-parallel form:

| Shape | dtype | per-row GB/s | streaming GB/s |
| --- | --- | ---: | ---: |
| `ffn_gate` 2560 -> 9216 | Q4_K | 40.5 | 286.6 |
| `ffn_down` 9216 -> 2560 | Q6_K | 81.6 | 256.1 |
| `ffn_down` 9216 -> 2560 | Q4_K | 43.6 | 192.6 |
| `attn_q` 2560 -> 8192 | Q4_K | 40.6 | 258.2 |
| `attn_qkv` 2560 -> 8192 | Q5_K | 43.1 | 248.6 |
| `attn_gate` 2560 -> 4096 | Q4_K | 39.9 | 254.8 |
| `attn_output` 4096 -> 2560 | Q4_K | 42.6 | 203.2 |
| `ssm_out` 4096 -> 2560 | Q5_K | 44.2 | 203.4 |
| `attn_k` 2560 -> 1024 | Q4_K | 30.3 | 90.4 |
| `attn_v` 2560 -> 1024 | Q6_K | 51.4 | 136.7 |
| `ssm_beta` 2560 -> 32 | Q8_0 | 26.9 | 5.0 |
| `lm_head` 2560 -> 248320 | Q6_K | 46.5 | 184.2 |

The three cells below 150 GB/s are the 1024-output attention projections: a
32-block grid cannot fill a 40-compute-unit device, and one such submission costs
16 microseconds, so the decode step spends 0.26 ms of its 33.4 ms on the sixteen
of them. Closing that needs split-K or fewer channels per block, which is not
taken yet.

Inside the decode step the same kernels run at 94-187 GB/s rather than the
isolated figures, because a token rotates through 2.7 GB of weights and every
matrix is cold in the 32 MB MALL ([benchmarks.md](benchmarks.md#decode-matvec)).

### 5.4 Q4_0 and Q5_0

Q4_0 and Q5_0 use 32-value blocks. Q4_0 stores one FP16 scale and sixteen packed
nibble bytes; values are `d * (q - 8)`. Q5_0 adds a four-byte high-bit plane;
values are `d * (q - 16)`. Both are supported by the CPU reference, single and
batched HIP GEMV, embedding, and WMMA prefill.

### 5.5 IQ2 and IQ3 reference support

The CPU reference dequantizes `IQ2_XXS`, `IQ2_S`, and `IQ3_S` blocks with the
ggml codebooks and sign tables. `IQ2_XXS` and `IQ3_S` derive sub-block scales
from packed auxiliary words; `IQ2_S` uses one 4-bit scale pair per 32 values.
`tensor.DequantizeRow` and `tensor.Matrix.MatVec` use the same block decoders,
so the reference path never expands a whole model to FP16. On the device the
same three codebooks live in `kernels/hip/iq_decode.cuh` as `__constant__`
tables and per-value decoders. Only two kernel families read them: the MoE
expert kernels, and `dequant_span_kernel`, which backs `yuuka_dequant_f16`.
General GEMV, GEMM, and LM-head validation still rejects these dtypes, so an IQ
tensor outside the MoE family is a planning error rather than a silent zero.
`Q4_1`, `Q5_1`, `Q2_K`, `Q3_K`, `Q8_K`, `IQ1`, `IQ2_XS`, `IQ3_XXS`, `IQ4_NL`,
and `IQ4_XS` remain capability failures.

### 5.6 Precision

For FP16/BF16 inputs, accumulate in FP32 unless a measured, documented
tradeoff justifies otherwise. Quantization scales and mins must be applied
before the final rounding step; avoid double rounding between unpack and
accumulate. Current compute kernels write FP32 outputs. An output-dtype plan
field must be added before FP16 outputs are exposed.

## 6. Prefill: WMMA and quantized GEMM

### 6.1 Implementations

Prefill uses a separate operator family from decode. The runtime selects one of:

- **Batched GEMV**: the Phase 2 kernel, limited to 16 rows. It is retained as
  the baseline for very small prompt chunks.
- **rocWMMA GEMM**: the custom kernel in `kernels/hip/gemm/wmma_gemm.hip`. It
  accepts FP32 activations and every supported weight dtype.
- **rocBLAS FP16**: the external baseline in `kernels/hip/gemm/gemm_blas.hip`.
  It requires an FP16 activation buffer; the planner inserts an explicit cast.
- **Int8 MMQ**: `kernels/hip/gemm/mmq_gemm.hip` keeps Q4_K/Q5_K/Q6_K weights
  quantized, stages both operands in WMMA fragment order, quantizes the
  activations to Q8_1, and accumulates exact int32
  products in 16x16x16 int8 WMMA. See 6.3.

All four are validated against a float reference built from the exact packed
weights. The rocBLAS baseline is a comparison point and a dispatch choice for
the measured F16 shape classes, not a fallback for unsupported quantization.

### 6.2 Custom WMMA GEMM

Two rocWMMA tile variants share the packed-load helpers and are selected inside
`launch_gemm_wmma_f32out` from the measured row and shape class:

- `gemm_wmma_k32_kernel`: 32x64 output tile, eight waves, one 16x16 fragment
  per wave, 32-wide K step. This is the original Phase 3 kernel.
- `gemm_wmma_kernel<DType, 64, 2, 32>`: 64x64 output tile, eight waves, one A
  fragment and two B fragments per wave, 32-wide K step with double-buffered
  shared-memory staging. One decoded weight tile feeds two accumulator
  fragments and 64 activation rows.

Both variants:

1. Load the FP32 `X` tile and convert it to FP16 in shared memory.
2. Stream and decode the weight tile directly into the FP16 shared tile, one
   contiguous span per thread.
3. Accumulate in FP32 with 16x16x16 WMMA operations.
4. Stage the FP32 accumulator tile in shared memory and write only valid
   output elements, so non-aligned `M` and `N` edges are safe.

`load_b_tile` gives each of the 64 weight rows four threads, so one thread
decodes a contiguous span of 8 values at a 32-wide K step (16 at a 64-wide
step). A span stays inside one 32-value quantization segment, so the block
header, scale, and min are loaded once per span. Q4_K and Q5_K read their
packed values with `uint2`/`uint4` global loads: their row byte sizes are
multiples of 16 and the span offsets are 8/16-byte aligned by construction.
Decoded halves are written straight to the shared tile, so no span is staged in
registers or scratch. An earlier version that buffered the span in a local
array spilled 48 B/lane to scratch and lost 6-12% on the wide 512-row shapes,
so the direct shared-memory write is the accepted form. Q4_0, Q5_0, and Q8_0
decode the span with scalar loads.

Q6_K decodes four values per thread from one `ql` word and one `qh` word
(`q6k_span4`), which retires the per-value byte loads of the element mapping.
The group shares one scale byte and one block header, and its four values always
sit in one 32-value segment, so the 16-bit halves of the two words are all the
block reads it needs. A block is 210 bytes, so the words are assembled from two
16-bit loads: a 32-bit load would be undefined-aligned on the odd blocks. Eight
lanes cover one 64-column row and a pass stages 32 rows. This replaced the
element mapping for Q6_K, which measured 6.115 ms on `9216 -> 2560` at `M = 512`
and 4.882 ms on `2560 -> 9216`; the span loader measures 3.589 ms and 3.175 ms
on the same cells, and the earlier span walk's 54% regression on small batches
does not recur because the span now amortizes the header over four values
instead of eight. The decoded values are bit-identical to `q6k_value`, so the
chunk gate did not move.

F32 and F16 keep the element mapping, in which consecutive lanes read
consecutive row bytes. They have no block header worth hoisting above the
per-value cost, and the span walk measured 12-16% slower for F16 on tall-K
shapes because a thread's bytes are then strided across rows instead of
coalesced across lanes. This kernel never materializes a dequantized weight
matrix: every supported dtype shares the tile path and differs only in the
packed-load helper. The separate rocBLAS route does
materialize one FP16 weight tile per call, in a caller-owned scratch sized from
the largest quantized matrix, and it is used only for the wide K-quant class
described below.

The FP32 accumulator is a measured requirement, not a default. FP16 compute
(after dequantizing into an FP16 scratch) is 1.6x faster in the library but
moved the 512-token prefill logits by 0.305 against the 0.05 chunk gate, so the
wide K-quant class instead runs rocBLAS with FP32 compute and an FP16
destination: 22.0 TFLOP/s on `2560 -> 9216` at `M = 512` against 8.6 for this
kernel, with one rounding of each output instead of one per K step. Tall-`K`
shapes keep this kernel because rocBLAS splits `K` there and the FP16
destination forces that reduction through FP16 (1.1-1.5e-3 measured against
4.0-4.9e-4 on the wide shapes). A 128-row tile with four accumulator fragments
was also measured: it halves the weight-tile reloads but needs a two-half
epilogue for the 24 KiB staging alias, and it loses on tall-`K` shapes where the
extra CTA count matters more. All of these are recorded in
[benchmarks.md](benchmarks.md#decisions-and-rejected-routes).

The 64x64/K32 variant is selected for `rows >= 64` when `K <= N`; the 32x64
variant remains selected for tall-K projections and smaller batches. Tall-K was
re-measured with the span loader instead of assumed: at `M = 64` and `M = 128`
the 64-row tile costs 40-90% on F16 and Q6_K tall-K and about 10% on Q4_K,
because halving the output columns per CTA doubles the number of CTAs that hide
the long K loop. At `M = 512` tall-K the two tiles are within 10%, so the 32x64
tile stays for every `K > N` shape. A 64-wide K step was also measured for the
64x64 tile and is 15-30% slower than the 32-wide step with the span loader
(Q4_K 2560 -> 9216 at `M = 512`: 3.3 ms against 2.85 ms), so only `BlockK = 32`
is instantiated.

The 64x64/K32 kernel stages 16 KiB, so four CTAs fit the 64 KiB LDS of a CU
instead of two for the previous 32 KiB, 64-wide staging, and reports 84-86
VGPRs, 28-48 SGPRs, no spills, no scratch, and 16 waves/SIMD. The 32x64 variant
reports 42-43 VGPRs, 25-46 SGPRs, 14 KiB static LDS, no spills, and 16
waves/SIMD. Measured on 2026-11-16 with the same binary pair, `M = 512` shapes
improve 48-50% for Q4_K, 49-54% for Q5_K, 32-49% for Q8_0, 3-23% for Q6_K, and
14-27% for F16, and the 512-token prefill moved from 544 to 881 tok/s at the
time of that change. See [benchmarks.md](benchmarks.md#isolated-kernels).

The int8 MMQ route (6.3) now carries K-quants from 128 rows up, so the current
numbers for the wide shapes come from it: 17.3-18.3 TFLOP/s on Q4_K
`2560 -> 9216` at `M = 512` against 8.4 for this WMMA tile, and a 512-token
prefill of 1612 tok/s end to end against llama.cpp's 2346 tok/s on the same
device. The gap record is in
[benchmarks.md](benchmarks.md#llamacpp-comparison).

A CTA-order experiment that mapped consecutive blocks to M tiles was measured
and rejected: it reduced the tall-K time but raised the wide-projection time,
and the position-0 chunk GEMM total regressed to 947 ms. The kernels keep `N`
fastest in `blockIdx.x`.

`gemm_wmma_swiglu_kernel` fuses the FFN gate and up projections. It stages one
activation tile and both weight tiles, accumulates with two pairs of fragments,
and applies `silu(gate) * up` to the accumulator elements before the single
output store. Both weights must share dtype and shape. The fused kernel uses a
32-wide K step (24 KiB shared memory for the 64-row tile) because a 64-wide step
needed 48 KiB and was 72% slower for the fused pair. The plan selects it only
when both projections resolve to `PrefillWMMA`; otherwise the separate gate,
up, and `SILU_MUL` operations remain. Q/K/V, beta/alpha, and RMSNorm fusions are
not implemented.

Larger tiles, an int8 tensor-core path, and the remaining projection fusions
remain candidates; they are not implemented and
are not claimed in the benchmark results.

### 6.3 Int8 MMQ GEMM

`mmq_gemm.hip` follows llama.cpp's MMQ design at the algorithm level (MIT,
commit `b1b770d90`); staging, tile shape, scale layout, and epilogue are an
independent implementation validated against the CPU reference in
`tests/hip/prefill_test.cpp`.

Activation quantization (`quantize_q8_1_kernel`, public ABI
`yuuka_quantize_q8_1`, plan op `YUUKA_OP_QUANT_Q8_1`): one wave handles 128
contiguous values, four per lane. Each 32-value block keeps `d = amax/127` and
`d * sum(q)`, so a Q4_K/Q5_K weight minimum becomes a dot product with the
activation block sum. Ties round to even (`rintf`). The width must be a multiple
of 128, so a wave never crosses a row.

The generic `gemm_mmq_kernel` computes a 64 output-channel by 128 token tile with
eight waves, a 128-value K step, and 30720 bytes of dynamic shared memory:

| Slice | Layout | Bytes |
| --- | --- | ---: |
| A, decoded weights | fragment order, `[4][4][512]` | 8192 |
| B, Q8_1 activations | fragment order, `[8][4][512]` | 16384 |
| A scales | `[8][64]` float, scale-slot major | 2048 |
| B scales | `[4][128]` float2, block major | 4096 |

Each wave owns one 16-row weight fragment at `row_base = (warp % 4) * 16` and
four 16-column activation fragments at `col_base = (warp / 4) * 64`. The int32
accumulator mapping is fixed by the builtin and was measured against the CPU
product on this device: lane `l`, element `e` holds output row
`2*e + l/16` of column `l%16`, so a lane applies one activation scale (its
column) and eight weight scales (its rows) per block.

Both operands are staged in the fragment register order of the int8 WMMA, not in
row-major tiles. The Q8_1 buffer itself stays row-major, so the public quantizer
contract and the digit check in `tests/hip/prefill_test.cpp` are unchanged; the
swizzle happens in the LDS copy of `stage_activations`. A 16x16 int8 fragment is 256 bytes and `rocwmma` loads eight
bytes per lane, where lane `l` holds row `l%16` of the 8-value K half `l/16`.
Staging therefore groups one 32-value K pair into a 512-byte region whose lane
`l` slot holds 16 bytes: the first 16-value slice in bytes 0-7 and the second in
bytes 8-15. One `ds_read_b128` then feeds two MMAs, five 128-bit loads serve the
eight MMAs of a K pair, and the staging stores become 16-byte lane stores.

Staging is assigned per warp: each warp owns whole 512-byte regions and every
lane fills one 16-byte slot, so a warp store covers one contiguous region. The
measured `LDSBankConflict` rate is 0.0% against 35.9% for the previous
272-byte-stride row-major staging. Q4_K/Q5_K spans are decoded with a
word-wide nibble expansion (a shift and one `0x0F0F0F0F` mask for eight
digits, plus a bit-plane move for the Q5_K fifth bit); Q6_K block starts are only
2-byte aligned, so its spans are still assembled from 16-bit halves.

The K step is 128 values rather than 256. The staged operands then take 30 KiB,
which keeps two blocks resident per CU (measured `OccupancyPercent` 45.6%
against 23.9%), and the staging stalls of one block are covered by the other
block's MMAs. A 256-value step needs 60 KiB and one block, and a 64-value step
adds enough barriers to lose the tall-K shapes; both were measured and rejected
(see [benchmarks.md](benchmarks.md#decisions-and-rejected-routes)).

The Q4_K/Q5_K scale (one per 32 values) is applied after each pair of 16-wide
slices, which is exact because the int32 partial sum is exact and the scale is
constant across the block; the minimum term adds `-(d_m * m) * d_a * sum(q)`.
Q6_K has one scale per 16 values and no minimum, so it applies the scale per
slice. The weight scales load once per block and are reused by all four
accumulator tiles.

The epilogue writes the accumulators straight to global memory. For a fixed
fragment the warp's 16 low lanes cover 16 consecutive tokens of one output
channel, so each store instruction covers two 64-byte runs. Transposing through
a shared-memory tile instead needs 33 KiB, which would break the two-block
residency, and the direct stores measure the same within noise.

The generic loop issued 22.2 VALU and 4.8 LDS instructions per WMMA on
Q4_K `2560 -> 9216` at `M=512`. Two costs dominate. First, rocWMMA duplicates
gfx11 wave32 operands in registers, adding 2.5 `ds_swizzle_b32` instructions
per WMMA on the LDS pipe. An earlier experiment staged the duplicates
explicitly, raising total LDS use to 51 KiB and cutting residency to one
block per CU.
Second, the scale application is four VALU per accumulator element per block
(`int32 -> float` conversion, the scale product, the main FMA, and the minimum
FMA), which is 16 VALU per WMMA by construction: each block updates eight
elements per fragment while issuing two MMAs per fragment. `v_fma_mix_f32` does
not convert an integer operand on gfx1151, so the conversion cannot be folded.

Tails are masked: rows outside the batch stage zeros (their scales and sums are
zero) and output rows and channels outside `n_out` are not written. `n_in` must
be a multiple of 128; every K-quant row layout is a multiple of 256.

Earlier generic kernel resources on gfx1151: 133 VGPRs, 38-40 SGPRs, no spills, no scratch,
10 waves/SIMD resource limit with two blocks resident per CU, dynamic LDS
30720. The quantizer reports 20 VGPRs and 16 waves/SIMD.

Generic baseline throughput at `M = 512` (sweep, five timed submissions after two
warmups, including the Q8_1 pass): Q4_K 17.3-18.3, Q5_K 17.5-18.1, and Q6_K
12.4-14.4 TFLOP/s across the three shapes, against 13.5-14.1, 12.6-13.4, and
11.0-11.6 TFLOP/s before the layout and occupancy change. At this baseline,
the generic route ran 1.2–1.3× below llama.cpp's MMQ kernel on the same
device. [Measured attribution](benchmarks.md#prefill-gemm)

The q/k/v and qkv/gate groups read the same normalized tile, so the plan
quantizes it once per group instead of once per projection
(`prefillProjections` in `engine/gpu/plan.go`); a 512-row chunk runs 128
`quant_q8_1` operations instead of 168 before FFN activation fusion.

When gate, up and down all select MMQ, the FFN uses the same Q8_1 inputs
for gate and up, then reuses those buffers for down on the same stream.
The `SILU_MUL_QUANT_Q8_1` operation reads the FP32 gate and up
projections, applies SwiGLU and writes Q8_1 digits and block scales.
The combined prefill and decode ABI 9 assigns it opcode 43 and assigns
opcodes 38–42 to the fused decode operations. The fused quantizer
uses the plain quantizer's scale, rounding and sum rules and skips
materializing the activated FP32 tensor. The compact final FFN uses
its parent chunk's dispatch. [FFN validation and timing](prefill-optimization/acceptance/05-ffn-fusion.md)

#### gfx1151 direct MMQ specialization

The native launcher selects direct MMQ for the measured shapes in the
table, `M` in 128, 512 or 2048, and 16-byte aligned packed weights and
Q8_1 digits. gfx1151 compiles the direct WMMA path. Other device targets
compile the generic kernel. The direct tile covers 128 channels by 64 tokens.

| Weights | Input width | Output widths |
| --- | ---: | --- |
| Q4_K | 2560 | 1024, 4096, 8192, 9216 |
| Q4_K | 4096, 9216 | 2560 |
| Q5_K | 2560 | 8192 |
| Q5_K | 4096 | 2560 |

The original Q4_K FFN cell `M=512, K=2560, N=9216` uses its compile-time
dimensions. The other cells pass dimensions into the same tile algorithm.
Each half of the generic grid's x range covers the same 128-channel
tiles for a different 64-token half. The shared-memory allocation is 30 KiB.

Within each 512-byte operand region, the specialized LDS layout stores
two contiguous 16-value slices per row. Both halves of each wave load the
same operand from LDS. A staging thread reads packed weight bytes once
for two adjacent subblocks and writes both sets of digit slices into
LDS. Q5_K reads the shared fifth-bit words and extracts a separate bit
plane for each of the same subblocks. Each activation block uses
16-byte vector loads and stores. Its
staging thread writes the FP32 scale and scaled sum once. The kernel
completes both integer WMMA instructions and
FP32 scale/min correction of each fragment before consuming the next
fragment. [FFN cell](prefill-optimization/acceptance/02-q4-ffn.md) ·
[Additional projections](prefill-optimization/acceptance/04a-mmq-shapes.md)

### 6.4 Dispatch

Dispatch is based on `M`, `N`, `K`, quantization type, and the measured row
class. The table is in `backend/hip.SelectPrefillPath`; measurements are in
[benchmarks.md](benchmarks.md).

| Condition | Selected path |
| --- | --- |
| `M < 4` | Batched GEMV |
| Quantized, `M < 8`, and `K > N` | Batched GEMV |
| K-quant, `N >= 64`, `M >= 128` | Int8 MMQ (Q8_1 activations, int32 WMMA) |
| Quantized, other `M >= 4` | Custom rocWMMA |
| F16, `4 <= M <= 64`, `K > N` | rocBLAS + cast |
| F16, `M < 8`, `K <= N` | Batched GEMV |
| F16, `M >= 128`, `K <= N` | rocBLAS + cast |
| F16, other `M >= 8` | Custom rocWMMA |

The custom kernel is selected only where the sweep showed it faster than the
compared baseline, and a library route only where it is both faster and inside
the numeric gates. The K-quant MMQ row is bounded from below by the Q8_1 pass
and the 128-token tile (it loses to WMMA below 128 rows) and requires
`N >= 64` because narrower output tiles waste more than half the tile. It
replaces the dequantize-plus-rocBLAS route for K-quants entirely: that route was
2-17% faster on wide shapes at 512 rows but is numerically rejected on tall
shapes (see 6.2), and it needed an FP16 copy of the weight matrix per call. F16
baselines include the cast cost. A dispatch table entry
is valid only for the recorded gfx1151 environment; the kernel is not selected
from assumptions about other RDNA parts.

The prefill tail (the output projection over the last chunk row, or over the
candidate rows of a speculative block) is not on this table: it is a `N >> M`
shape with `M <= 16`, where every route above wastes the tile on rows that do
not exist. It runs `lm_head_stream` instead (6.5) whenever the dtype is
supported, `n_in` is a multiple of 256, `M` is in `1..16`, and
`N >= 1024`; anything else keeps one decode GEMV per row. The decode path itself
is unaffected: it always uses the GEMV family.

Weight reuse is a secondary effect, not the binding limit. A 32-row M tile
makes a 512-row chunk read every weight tile 16 times, and the 64-row tile
halves that, but halving it again with the measured 128-row tile only bought
6-26% on wide shapes and lost 0.5-15% elsewhere. The kernel is instruction-issue
bound: 114 instructions per 4096-MAC WMMA, of which 54 are VALU and 10 LDS.
The 2026-11-15 measurements put a 512-token prompt at 516 tok/s, the
2026-11-16 span-decoding build at 881 tok/s, and the int8 MMQ route now
measures 1324 tok/s on the same GGUF and device, against llama.cpp HIP at
2346 tok/s for the same 512-token prompt in the current reference build. The
2026-11-16 split was matmul 293 ms, the gated delta-net chunk 50 ms, and
prefill attention 19 ms at position 0 and 562 ms at position 7680; attention at
position 7680 dropped from 2030 ms to 643 ms. The per-workstream measured status
is in [benchmarks.md](benchmarks.md#prefill-prompt-length).

### 6.5 Streaming LM head

For ordinary prefill ending in a full-attention layer, the plan puts that
layer's query, attention output and FFN in the output tail. All chunk
rows still write K/V. The final chunk compacts its last residual into
row zero and evaluates one query and FFN row using the parent chunk's
projection and attention routes. RoPE and paged attention use the
last prompt position via the ABI 7 token displacement. An intermediate
chunk ends after the final K/V write. Speculative verification computes
every requested row through attention and FFN.
[Validation and timings](prefill-optimization/acceptance/06-final-query.md)

`lm_head_kernel` in `kernels/hip/gemm/lm_head.hip` computes the output
projection in the prefill and decode tails. A decode matvec with one row and a
512-value K step takes the remapped one-row kernel of section 5.4 instead; the
channel-parallel form below serves every other row count and K width:

```text
out[m][n] = sum_k W[n][k] * x[m][k],   m in [0, rows), rows <= 16
```

`x` and `out` are row-major FP32 blocks, `W` is any packed dtype in the GEMV
family, and `n_in` must be a multiple of 256 so that every K step is one
quantization block. The kernel is not a tile GEMM: with `N` in the hundreds of
thousands and `M = 1` every tile form wastes 15 of 16 accumulator rows, and the
operator is pure weight bandwidth.

Mapping:

- One wave owns four output channels and walks the whole K axis itself. A lane
owns four consecutive K values of each of its channels, so 32 lanes cover a
128-value half without overlap and the packed reads of the wave are contiguous
32-byte sectors of the four arrays a Q6_K half touches (`ql`, `qh`, `scales`,
`d`).
- The activation row block for the current 256-value K step is staged in
dynamic shared memory once per workgroup, so the activations are read from L2
once per 32 output channels instead of once per channel. The bytes are
`rows * 256 * 4`, that is 1 KiB for the single-row tail and 16 KiB at the
16-row cap.
- `decode_span4` in `quant.cuh` decodes a lane's four packed values straight
into registers, with the block base, nibble selection, and K-quant scale and
minimum hoisted out of the four values. Each decoder restates the arithmetic of
the per-value decoder of the same dtype, so both produce the same digits; the
per-value form is 3.2x slower here because it recomputes the block arithmetic
for every value.
- `rows` is a compile-time bound, not a runtime loop bound: the single-row
instantiation uses 45-67 VGPRs and the 16-row one 97-127, where a runtime bound
indexes the accumulator array dynamically and spills 272 bytes per lane to
scratch (measured 19.9 GB/s against 88 GB/s).
- The reduction is one `shfl_xor` tree per channel per row, so a second
identical submission is bitwise identical.

Measured with the `LMHEAD` cells of `build/yuuka_prefill_sweep` (median of five
timed submissions after two warmups) on the recorded gfx1151 device, `n_in`
2560:

| dtype | N | M | per-row GEMV | streaming LM head |
| --- | ---: | ---: | ---: | ---: |
| Q4_K | 248320 | 1 | 15.01 ms (24 GB/s) | 2.36 ms (152 GB/s) |
| Q4_K | 248320 | 16 | 240.82 ms | 6.59 ms |
| Q6_K | 248320 | 1 | 11.94 ms (44 GB/s) | 2.90 ms (180 GB/s) |
| Q6_K | 248320 | 16 | 191.00 ms | 7.67 ms |
| Q8_0 | 248320 | 1 | 7.22 ms (93 GB/s) | 3.11 ms (217 GB/s) |
| Q8_0 | 248320 | 16 | 115.51 ms | 7.14 ms |

In the 512-row prefill plan of `TestGPUProfilePrefill` the tail costs 2.84-2.92 ms
of a 287 ms chunk, against 11.3 ms for the single-row GEMV tail before this
kernel. The kernel is
bandwidth bound: the same 521 MB Q6_K matrix streams at 244 GB/s in a pure read
loop on this device, so 2.90 ms is about 85% of that ceiling.

The decode tail runs the same single-row instantiation. In the profiled decode
step of `TestGPUProfileDecode` it costs 2.79 ms at 187 GB/s against 11.23 ms at
46 GB/s for the per-row GEMV, which is 8.4 ms of a 97.4 ms step.

Numerics. The per-element products are the same FP32 products the decode GEMV
forms, including the K-quant scale-then-minimum order; only the summation order
differs, because a lane accumulates four neighbours and the last five additions
are a shuffle tree. The opt-in int8 variant of the same kernel replaces the FP32
products with a `v_dot4` over Q8_1 digits: it moves decode logits by up to 0.2656
over eight greedy steps with identical tokens, against the 0.0031-0.264 band the
prefill int8 route records, and its kernel output matches a host reference that
uses the same digits to 5.7e-4 relative. `tests/hip/prefill_test.cpp` checks all eight dtypes against
the independent host dot products for one, three, and sixteen rows and requires
a repeated run to be bitwise identical; `TestPlanLMHeadMatchesDequant` checks
Q6_K and Q8_0 against dequantized CPU dots (Q8_0 is bitwise equal), and the
model-level gates are unchanged (`TestGPUMatchesCPU` 0.00309944,
`TestGPUChunkedPrefillLongContext` top-1 `33075` with zero difference).

A shape the kernel does not cover keeps the per-row GEMV tail; nothing silently
changes numerically. `backend/hip.SelectLMHeadPath` is the one rule both tails
use: dtype in the GEMV family, `n_in` a multiple of 256, `n_out` at or above
`lmHeadMinOutputs`, and 1-16 rows.

The crossover was re-measured at the decode width (`n_in = 2560`, one row) with
`build/yuuka_gemv_decode_sweep --threshold`, five warmups and twenty-five timed
submissions per cell. The streaming kernel wins from 512 outputs upward for
Q4_K, Q5_K, and Q6_K, and from 1536 upward for Q8_0 and F16, where the per-row
GEMV's vectorized path is still ahead at 1024 (0.022 ms against 0.035 ms). The
threshold stays at 1024 because one number covers every dtype without
regressing Q8_0, which is level there, and because no model shape sits in the
512-1536 range: an output projection is vocabulary sized. The table is in
[benchmarks.md](benchmarks.md#decode-matvec).

Numerics of the decode route. The per-row GEMV sums a strided set of K values
per thread and reduces over the block; the streaming kernel sums four
neighbours per lane and reduces across the wave. Both accumulate in FP32, and
`TestGPUStreamingTailMatchesGemv` measures the difference on the real model:
4.8e-06 maximum absolute logit difference over eight greedy decode steps, with
identical tokens.

## 7. Attention

### 7.1 Prefill: Flash Attention
`attention_prefill_f32_kernel` (and its paged counterpart) processes a prompt
chunk with one wave per query row. A workgroup owns 16 query rows for one head
with 512 threads.

- Q is read into registers, with each lane holding `ceil(head_dim/32)` values.
- K and V are staged together, 16 keys at a time, in dynamic shared memory as
  FP32. The shared layout stores each lane's `vec` values with a stride of
  `vec + 1`, so the lane-parallel dot-product and value reads are bank-conflict
  free. The workgroup reuses one staged V tile across all 16 query rows, which
  cuts global V traffic by 16x relative to the previous one-row tile. Query
  heads that share a KV head still stage their own tile; they are not grouped
  in one workgroup, and only L2 reuse applies across them.
- Scores are reduced within the wave; online softmax keeps per-row max, sum,
  and FP32 value accumulators, so the full attention matrix is never
  materialized.
- Causality uses the chunk position: query row `r` may attend to keys
  `0..position+r`. Waves in a workgroup share the K/V-block loop to keep
  `__syncthreads` convergent and mask keys beyond their own position.
- GQA maps query head `h` to KV head `h / (heads/kv_heads)`.
- The Qwen3.5 attention gate (the second half of each query vector) is applied
  as `sigmoid(gate)` before the output write. The contiguous op implements only
  this gated layout; `ATTENTION_PREFILL` rejects `u[3] != 0` with an unsupported
  capability error. The dense query-layout flag is retained by the paged
  variants described in section 7.4.

Dynamic shared memory is `2 * 16 * 32 * (vec + 1) * 4` bytes, or 36 KiB at
`head_dim = 256`. The kernels report 49-53 VGPRs, 48-63 SGPRs, no spills, and
16 waves/SIMD. F16 KV is still loaded and staged as FP32; with the staged-V
kernel it measured slightly slower than F32 KV at position 7680 (657.5 ms
versus 643.1 ms). F16 is the project default; F32 remains available for the
reference path and for that long-context tradeoff.
The measured position-7680 attention time is 643 ms with F32 KV, down from
2030 ms before this change.

Supported prefill head dimensions are 1-256; the plane-stage kernel reports an
unsupported capability error above 256. Decode attention uses the split-key
kernel described in section 7.2.

Chunked prefill splits a prompt into at most `chunk-size` rows (default 512,
maximum 4096). KV writes for a chunk happen before its attention, so later
chunks see the full causal prefix. The runtime builds and caches one plan per
chunk length; only the final chunk executes the output tail.

### 7.2 Decode: Split-Key Paged Attention

`attention_paged_flash_kernel` in `kernels/hip/attention/attention.hip` is the
decode attention kernel. It splits the key range so the cost follows the KV
bytes instead of the position count:

- Grid `(kv_heads, ntiles)`. One block owns one KV head and one contiguous tile
  of key positions, and all `heads / kv_heads` query heads that share that KV
  head. A KV row is therefore read once per block rather than once per query
  head; on Qwen3.5 that is four query heads per block.
- The tile is 32 positions up to a 1024-position capacity and 64 above it, so a
  short context gets more blocks and a long context writes fewer partials. The
  tile length is a kernel argument and the shared-memory arrays are sized for
  64.
- Score phase: each warp owns one position of the tile, reads the lane's part of
  the K row (16 bytes of F32, 8 bytes of F16, one Q8_0 group per lane), and
  reduces the dot products for every query head of the block across the wave.
- Online softmax: one thread per query head reduces the tile maximum, every
  thread rescales its running maximum, normalizer, and accumulators, one warp
  per head exponentiates the tile, and one warp shuffles the tile weight sum.
  One barrier per phase, three per tile.
- Value phase: each thread owns two output dims, so a warp covers a 32-value
  span of the V row; the tile weights come from shared memory as one vector
  load per position.
- Each logical position `p` maps to
  `block_table[p / block_size] * block_size + p % block_size`; K and V are read
  from the pool through that row. The block table and pool are device buffers.
- The block writes one partial output vector per head and tile plus its running
  maximum and normalizer. `attention_flash_combine_kernel` merges them with the
  log-sum-exp rescaling, one 64-dim slice of one head per block, and applies the
  attention gate as `sigmoid(gate)` before the output write.
- The dynamic variant reads `position` from a device parameter buffer for both
  the split kernel and the merge, so a captured HIP Graph replays without
  changing kernel arguments.

The scores workspace is the partials arena: `heads * ntiles * (head_dim + 2)`
floats with `ntiles = ceil(capacity / tile)`, the outputs first and then two
floats per head and tile. `decodeAttentionScratch` in `engine/gpu/model.go`
allocates it and must agree with `flash_tile` in the kernel.

Shapes outside the tiled kernel's template set keep the serial reference
implementation `attention_paged_f32_kernel`, one block per query head with a
sequential position loop: head dimensions above 256, head dimensions not a
multiple of four, and `heads / kv_heads` outside 1, 2, 4, 8. It is also the
byte-for-byte reference the tiled kernel is compared against.

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
K and V through the same row format. The split-key decode kernel reads one
32-value Q8_0 group, one 8-byte F16 span, or one 16-byte F32 span per lane in
the score phase, and a 4-value Q8_0 group, 4-byte F16 pair, or 8-byte F32 pair
per lane in the value phase, so all three formats share one mapping. Prefix-cache
keys include the format so cached blocks are never reused across formats.

### 7.4 Attention query layout

The paged attention kernels assume the Qwen3.5 query layout with a sigmoid
output gate in the second head-sized half. The plan carries a dense flag in
`u[3]` that skips the gate and uses `heads * head_dim` query stride; the ABI
retains that variant, but no supported model selects it. The `qwen35` path uses
the gated layout exclusively.

### 7.5 FP16 MMA paged prefill attention

`attention_prefill_paged_wmma_kernel` in
`kernels/hip/prefill/attention_wmma.hip` is the default prefill attention
kernel. `launch_attention_prefill_paged_f32` selects it when all of the
following hold, and falls back to the tiled per-head kernel otherwise:

- the pool is F32 or F16, so an FP16 tile staging is either lossless or a
  documented rounding step (Q8_0 stays on the tiled kernel);
- `heads == 4 * kv_heads`, so one workgroup can serve the four query heads that
  share a KV head;
- `head_dim` is 64, 128, or 256, the widths the kernel is instantiated for;
- `rows >= 64` or `position + rows >= 256`: below that the chunk offers too few
  16-row query tiles to fill 40 CUs, and the tiled kernel measures faster.

Geometry. A workgroup is four waves, one per query head of the group, and owns a
16-row query tile. Each wave keeps the whole `16 x head_dim` output block in
accumulator registers (128 registers per lane at head width 256), so the value
product never leaves the wave. K and V tiles of 16 keys are staged in shared
memory once per workgroup and reused by all four heads.

Score orientation. The score product runs as `S^T = K Q^T` rather than
`S = Q K^T`, so both operands are row major in shared memory: K keeps the
key-row layout it is gathered in, and Q is stored transposed once per workgroup
instead of once per key tile. A score lane then owns one query row and eight
keys, so a row statistic is eight register operations and one cross-lane step
instead of a four-step butterfly, and the running maximum and sum are scalars
rather than per-row arrays. The transposed P the value product needs is written
back to shared memory from the same lane mapping.

Online softmax. The mask is applied inside the tile: a lane's element is `-inf`
when its key is beyond the query row or beyond the tile's causal bound. The
softmax runs in base-2 units, so an exponent is one `v_exp_f32` after a scaled
multiply, and the accumulator is rescaled by the row factor before each product.
The running statistics stay FP32, the accumulator is FP32, and only the staged
Q/K/V tiles and the probability tile are FP16.

Shared memory. At head width 256 the workgroup uses `(4*16 + 2*16)*256` halves
for the transposed Q and the K/V tiles plus four 16x16 probability tiles, 50 KiB
in total, which keeps two workgroups resident per CU. The kernel declares
`__launch_bounds__(128, 2)` so the compiler can use the full register file for
the accumulator block: at 192 registers it spilled 21 slots into the key loop,
and removing that spill was most of the 1.5x the stage measured.

Crossover and measurements. Isolated prefill attention, one layer, FP32 pool,
median of 20 launches after a warmup (full table in
[benchmarks.md](benchmarks.md#prefill-attention)):

| Shape | Tiled | MMA |
| --- | ---: | ---: |
| 16 rows at position 0 | 0.016 ms | 0.039 ms |
| 48 rows at position 0 | 0.044 ms | 0.052 ms |
| 64 rows at position 0 | 0.075 ms | 0.057 ms |
| 512 rows at position 0 | 2.468 ms | 0.507 ms |
| 32 rows at position 7680 | 14.651 ms | 3.266 ms |
| 512 rows at position 7680 | 70.243 ms | 14.815 ms |

A 512-token chunk therefore spends 4.1 ms in attention at position 0 and
117.6 ms at position 7680 over the eight full-attention layers, against 19.7 ms
and 562 ms for the tiled kernel.

Numerics. Staging rounds Q, K, and V to FP16, so an F32 pool gains the rounding
step an F16 pool already has. The native test measures 9.3e-5 maximum absolute
difference against the FP32 reference and against the tiled kernel on the same
plan input at head width 256, and the output is bitwise reproducible on a second
run. The full-model differential test records 0.015 maximum logit difference at
a 64-token prompt with the top-1 token unchanged; that is the same magnitude as
changing the KV storage format from F32 to F16 on the tiled kernel (0.011 on the
same prompt), which is the accuracy regime the current route runs in.

#### Split-key F16 prefill

On gfx1151 the gated F16 path with 256-wide heads and four query heads per
KV head supports a four-way key split for 512, 1024 and 2048 query rows in
the measured window from position 6144 through token 8191. The executor
uses the current position with the cached plan. Short prefixes and the
single-query output tail use direct attention.

A score-only pass over the leading 512 keys supplies a starting maximum
for the later ranges. Each range keeps its own online FP32 maximum, sum
and value accumulator. A wave merges the records for each row and head,
then applies the output gate. ABI 8 binds the optional FP32 records and
prefix maxima in slot 6. The plan reuses this scratch across layers on
the same stream. [Split-key acceptance](prefill-optimization/acceptance/07-long-attention.md)

The earlier group-of-four tiled kernel (`attention_prefill_paged_gqa_f32_kernel`)
is removed: the MMA kernel serves the same group at every shape the tiled kernel
won on, so keeping it would have left a slower path behind a silent condition.

Current limits:

- Q8_0 KV is supported but not faster than F32 at short context; its benefit is
  cache memory.
- The tiled fallback runs one workgroup per head and is
  6.5%-of-peak-issue bound on FP32 scalar per-lane arithmetic, which is why
  Q8_0 pools and short chunks keep the long-context gap.
- Direct MMA uses 16-row query tiles. Split-key MMA adds parallel workgroups
  over key ranges with the same query tile.
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
bins `(row, slot)` assignments into one bucket per expert using atomics. One
warp handles one row: every lane holds every 32nd expert in registers
(`moe_route_kernel<kPerLane>`, instantiated for 1, 2, 4, 8, 16, and 32 experts
per lane, so up to 1024 experts), a slot is a butterfly max with a lowest-index
tie-break that matches the ascending scan of the reference selection, and the
winner is masked out of the lane register set. A slot therefore costs one pass
over the row instead of a scan per slot. An expert count above 1024 fails plan
construction instead of selecting from a truncated candidate set.

The grouped kernels are the prefill form. They index blocks by
`(expert, 8 output rows)` and each warp owns one output row:

- `moe_gate_up_grouped` walks the expert's assignment list in tiles of eight
  (`kAssignTileIQ`, four for the general dtypes, whose scalar decoder needs the
  registers), decodes the gate and up span of the current K step once, and
  reuses that span across the whole tile. The stage therefore costs one
  weight-row decode per expert row per tile, not one per assignment, and the
  activation span of every assignment is loaded once per tile.
- SwiGLU uses the existing fused `silu_mul` kernel.
- `moe_down_grouped` performs the expert down projection over the same buckets
  with the same tile.
- `moe_reduce_add` applies routing weights and accumulates into the residual
  stream.

A K step covers `kStepK = 256` values, of which each lane decodes the eight
values at `base + 8 * lane`. `moe_span8` decodes those eight values with one
codebook fetch (two for IQ3_S, which holds four-value entries) instead of one
fetch per value; `iq2xxs_span8`, `iq2s_span8`, and `iq3s_span8` in
`kernels/hip/iq_decode.cuh` hoist the block offset, the codebook index, and the
sign word out of the per-value arithmetic, and produce exactly the values the
per-value forms produce. A warp reduces its eight lane partials with
`warp_reduce_sum`, so no block barrier and no shared-memory staging appear
anywhere in the expert path, and the reduction cost no longer scales with the
thread count. Spans that would cross the end of the row, and dtypes without a
span decoder, fall back to `moe_packed_value` per value.

The routing and assignment buffers are sized `experts * rows * used`, which is
safe because one row contributes exactly `used` assignments. A malformed plan
that would exceed a bucket is rejected at plan construction.

**The Q8_1 expert route.** The FP32 kernels above still pay one int8-to-float
conversion, one sign select, and one scale multiply per expert weight value,
which at these shapes is 805 M values per layer. For IQ2_XXS, IQ2_S, and IQ3_S
experts, a chunk of at least sixteen rows whose `hidden` and expert FFN width are
both multiples of 256 takes a second grouped family that keeps the digits
integral (`moe_gate_up_q8`, `moe_down_q8`):

- `quant_q8_1` quantizes the normalized tile once per layer into int8 digits plus
  one FP32 scale per 32 values, and `silu_mul_quant_q8_1` does the same for the
  expert SwiGLU product;
- each block stages sixteen assignment rows of that tile in shared memory for a
  256-value K step (264-byte row stride, so neighbouring rows miss the same
  banks) and decodes the gate and up span of the step into eight signed digits
  with `iq2xxs_dot8`, `iq2s_dot8`, or `iq3s_dot8`;
- one warp owns one output row and eight warps form a block, so one decoded
  weight span serves all sixteen assignments: an assignment costs one eight-byte
  LDS read, two `v_dot4` instructions, and two multiply-adds per eight values,
  and the two block scales are applied once per span instead of once per value.

The int8 product is exact, so the only difference from the FP32 route is where
the scales are applied. `tests/hip/iq_moe_test.cpp` runs the full Q8_1 pipeline
and requires it to match a host computation over the same decoded weights and the
dequantized Q8_1 activations to 2e-2; the measured difference is 1.5e-6, and the
two routes select the same experts. Chunks below sixteen rows, widths that do
not fill the tiles, and non-IQ experts keep the FP32 grouped kernels, which is
also what serves the cropped one-row FFN of the last layer; K-quant and IQ
experts take the int8 WMMA route described next when their shape fits its tile.
The route is a plan-build-time decision (`addMoEPrefillFFN`), not a runtime
option.

**The int8 WMMA expert route.** The Q8_1 route above still dots one assignment
at a time with `v_dot4`, so its cost grows with the assignment count even when
the weight bytes do not: the expert set is 9.06 GB at 128 and at 512 rows, and
the measured stage grows from 99.4 to 363.2 ms (91 to 24.9 GB/s of unique
weight bytes) while the reference implementation stays at the weight-read
regime. `moe_gate_up_mmq` and `moe_down_mmq` keep the same Q8_1 activation tile
and the same routed buckets, and replace the per-assignment dot with the int8
WMMA of the dense MMQ GEMM, staged through
`kernels/hip/gemm/mmq_tile.cuh`:

- One block owns one expert and one tile of output channels, and walks that
  expert's bucket in tiles of `n[5]` assignments (16, 32, 64, or 128), so one
  decoded weight tile serves the whole assignment tile. The bucket counts stay
  on the device, so the launch never synchronizes and stays graph capturable.
- The tile is 128 output channels at `n[5] <= 64` (one warp per 16-row fragment
  group, all eight warps busy) and 64 channels at `n[5] = 128` (two token halves
  of 64, four accumulator fragments per warp). The K step is 128 values, or 64
  at the widest assignment tile, which keeps the staged operands near the 30 KiB
  budget that holds two blocks per CU.
- The K-quant weight staging, the per-16/per-32 scale slots, and the Q4_K/Q5_K
  minimum term are the dense GEMM's, so the digits are the same ones the
  dequantized reference path sees. The activation tile is gathered through the
  routing arrays: the gate/up projections read the activation row of the
  assignment, the down projection reads the SwiGLU slot of the assignment, and
  both scatter the result to that slot row.
- The assignment tile width is chosen at plan build time from the measured table
  below, so one weight pass covers an expert's bucket at every measured chunk
  size.

The IQ expert codebooks fill the same tile through their own group staging,
`stage_iq_weights`: one thread owns one (row, 32-value group) task, fetches the
group header once, decodes all thirty-two values with `iq2xxs_dot32`,
`iq2s_dot32`, or `iq3s_dot32`, and writes both WMMA halves' 16-byte lane slots
plus the two scale entries of the group. The fragment layout is the dense GEMM's;
only the decode and the scale-slot rule differ per codebook. Each task loads the
group header once instead of once per half, which is what took the expert stage
from 24.9 to 61 GB/s of unique weight bytes at a 512-row chunk.

The dot32 forms apply the codebook signs branch-free: a set sign bit means the
value is negated, so the four digits packed in one 32-bit word are
`(v ^ mask) + (mask & 0x01010101)`, where `mask` has 0xFF in the negated bytes.
That is exact only because no codebook entry contains a zero digit, which the
grid tables guarantee (`tensor.TestIQGridDigitsAreNonZero`). The sign masks are a
compile-time expansion of the two sign encodings: IQ2_XXS indexes
`k_ksigns_iq2xs` with a seven-bit field and IQ2_S/IQ3_S use the byte and nibble
as the pattern. The staged digits are therefore the ones the dp4a route and the
FP32 reference path see.

| codebook | weight scale | `kSlicesPerScale` | minimum term |
| --- | --- | ---: | --- |
| Q4_K / Q5_K | one per 32 values | 2 | per-row minimum |
| Q6_K | one per 16 values | 1 | none |
| IQ2_XXS | one per 32 values | 2 | none |
| IQ2_S | one per 16 values (low/high nibble of `sc[ib32]`) | 1 | none |
| IQ3_S | one per 32 values | 2 | none |

`tests/hip/moe_mmq_test.cpp` pins the K-quant form for Q4_K and Q6_K experts,
whose scale layouts differ (per 32 values with a row minimum, and per 16 values
without), at assignment tiles 16, 32, 64, and 128, and at bucket sizes that do
not fill the tile. The projections are compared against a host computation over
the dequantized weights and the same Q8_1 values (measured 1e-6 relative), and
the down projection over the device SwiGLU tile.

`tests/hip/iq_moe_mmq_test.cpp` pins the IQ form for all three codebooks at the
same tile widths. The expert rows are built from the first block of each codebook
in the Qwen3.6 IQ2_XXS GGUF, and each bucket size (1, `J-1`, `J`, `J+1`, and
`3J+5` assignments, plus an empty bucket) is written directly into the routing
arrays, because a bucket that owns more than half of the assignments cannot come
from one top-k chunk. The gate and up projections are compared element by
element against the dp4a route over the same Q8_1 tile (measured 3e-8 relative),
which pins the digits, the scale slot, and the scatter; the down projection is
compared over the same Q8_1 SwiGLU tile (measured 1e-9 relative); and the whole
pipeline is compared against a host computation over the `yuuka_dequant_f16`
decode of the same blocks (2e-2). The IQ2_S rows are built with the two nibbles
of each group scale byte at opposite extremes (0 and 15, a 31x scale difference
between the two 16-value halves), so applying that scale per 32 values instead
of per 16 fails the test.

The route is the selected plan-build-time choice (`SelectMoEPrefillPath`) for
every dtype it decodes: the K-quants and IQ2_XXS/IQ2_S/IQ3_S, when every weight
row is a whole number of 256-value blocks and the output width fills the
128-channel tile. The assignment tile width comes from the measured table in
`moeAssignmentTile`: 16 up to 384 rows, 32 from 512 to 2048, and 64 from 4096.
The table is the measured winner of each row class in
[benchmarks/qwen3.6_35b_a3b.md](benchmarks/qwen3.6_35b_a3b.md#assignment-tile),
where the tile sweep and the end-to-end numbers are recorded; at 512 rows the two
widths tie in the sweep and the real model prefers 32 because its routing
concentrates more assignments per expert than a random draw.
`Options.ForceMoEQ8` keeps the Q8_1 dp4a route for the same shapes so the two can
be compared on identical plan inputs. On the real model the route reproduces the
golden four-token continuation and the chunk-invariant first token of the Q8_1
route (`TestGPUMoEMmqMatchesQ8`, `TestGPUChunkedPrefillLongContext`).

The grouped grid is the prefill form. At one token it would launch
`experts * ffn` blocks, of which only `used * ffn` correspond to selected
experts, so the decode plan uses a second family indexed by
`(output row, routing slot)`, with one warp per output row and the same span
decode and warp reduction:

- `moe_gate_up_selected` reads the routing id of the slot, decodes the gate and
  up rows of that expert, and writes `silu(gate) * up` for the slot.
- `moe_down_selected` reads the same routing id and forms the down projection of
  the slot's activation row.
- `moe_shared_add` computes `sigmoid(dot(gate, norm[row]))` and adds
  `s * src[row]` to the residual stream, which is the shared expert's gate in
  both the CPU reference and the HIP plan.

Both families decode expert weights with `moe_packed_value`, which accepts the
eight general matvec dtypes plus IQ2_XXS, IQ2_S, and IQ3_S. That helper is local
to the MoE kernels so `quant.cuh`'s `packed_value` keeps rejecting the IQ types,
and the launcher selects the compile-time codebook form (`MoeCodebook`) so the
IQ span decoders are reached without a per-element dtype switch.
`tests/hip/iq_moe_test.cpp` pins the three codebooks against the CPU reference
values and requires the grouped and selected pipelines to agree with a host
computation over the same decoded expert matrices.

Measured cost (2026-11-22, gfx1151, Qwen3.6-35B-A3B-UD-IQ2_XXS, tables and
reference comparison in
[benchmarks/qwen3.6_35b_a3b.md](benchmarks/qwen3.6_35b_a3b.md#measured-attribution)):

| Stage | Stage time | Share of its phase |
| --- | ---: | ---: |
| `moe_gate_up_mmq`, 512-row chunk | 93.8 ms | 22.2% of the chunk |
| `moe_down_mmq`, 512-row chunk | 49.1 ms | 11.6% of the chunk |
| `moe_gate_up_mmq` + `moe_down_mmq`, 128-row chunk | 58.6 ms | 42.1% of the chunk |
| `moe_route`, one row | 12 us per layer | 2.8% of a decode step |
| `moe_gate_up_selected` + `moe_down_selected`, one token | 3.6 ms | 21.0% of a decode step |

The routed expert stage reads the 9.06 GB expert weight set of a chunk once per
assignment tile. The int8 WMMA route decodes one 32-value codebook group per
thread into int8 digits and dots them with the int8 WMMA, so the stage streams
the set at 155 GB/s at a 128-row chunk and 64 GB/s at 512 rows, where the same
weight pass is followed by four times the assignment work (two token tiles of
MMA, four times the gathered activation bytes, and four times the expert-output
writes). At 128 rows the stage is within 1.2x of the reference's 49.7 ms for the
same two matrices; the record attributes the rest of the chunk.

The expert matrices remain the one tensor family this repository does not route
through `gemm_mmq` (`packed_value` rejects the IQ types, and
`dequant_span_kernel` would materialize an FP16 copy of 34 B parameters). Their
int8 route is the dedicated kernel pair above instead.

The prefill router projection is the one MoE matmul that does not follow the
measured prefill dispatch table: an F32 router takes the FP32 batched GEMM
(`launch_gemm_f32`) instead of the WMMA tile. Routing is a selection, so an
FP16-staged activation tile is enough noise to flip near-tie experts.
Perturbing the reference model's router weights by 1e-4 relative moves its
first-token logits by 0.64, and the WMMA router left the chunked prefill at
0.77 against the CPU reference where the FP32 route holds it at 0.009. The
projection is 2048x256 per row, so the accuracy costs nothing measurable. Every
other prefill matmul, including the shared expert, keeps the measured dispatch.

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
  128x128 recurrent update, and gated RMSNorm per head. The convolution is a
  plain `kernel-1`-tap FIR over the raw chunk, because the carry state holds
  only the previous raw inputs, so one thread per output element replaces a
  serial per-channel walk over the chunk; a second small kernel writes the
  trailing raw inputs for the next chunk. The convolution dropped from 26.4 ms
  to 8.7 ms per 512-row chunk, with the outputs unchanged because the tap
  summation order is the same. The register-resident path covers
  state size 128, where four lanes cooperate on one state column and each lane
  holds 32 of the 128 state rows in registers for the whole chunk. A layer
  exposes `heads * (state / 32)` workgroups. On gfx1151, for chunks of at
  least 128 rows whose FFN gate buffer fits the preparation span, a separate
  launch writes the normalized Q/K per token and key group, and the decay
  and beta per token and head. The core reads these FP32 arrays with no
  per-token block barriers. The FFN and recurrence use the scratch in stream
  order. The scratch slot is available from C ABI 5 and is mutually exclusive
  with the triangular scratch slot. Without either span, the register core
  recomputes Q/K per column workgroup using two block barriers per row.
  Both cores sum `S^T k` and `S^T q` across the four lanes with shuffles.
  Gated RMSNorm runs as a separate row kernel over each `(row, head)` pair,
  because splitting columns across workgroups would otherwise break the
  per-head normalization domain. State sizes other than 128 use the fused
  one-column-per-thread kernel, which reads and writes the state for every row.
  Both paths keep the row-major state layout shared with the decode kernel.
  [Prepared recurrence measurements](prefill-optimization/acceptance/04b-gdn.md)

  The lane split of the prepared core is a compile-time constant
  (`kGdnPreparedLanes`). Sweeping it from four lanes to eight and sixteen lanes
  — 64, 128, and 256 blocks of 128 threads for sixteen heads — moved nothing:
  61.2-61.8 ms for the 30 recurrent layers of a 512-row chunk at four and eight
  lanes and 62.2 ms at sixteen, against a 1.0 ms run-to-run spread of a single
  build. The core is limited by the per-row dependent FMA chain over the
  register-resident column and by the per-row `q`/`k`/`v` reads, not by the
  number of blocks, so the selected split stays at four lanes.

## 9. Gated delta-net

### 9.1 Chunkwise form

`kernels/hip/prefill/gdn_chunk.hip` implements the chunkwise WY form of the same
recurrence and is validated against `gdn_chunk_splitreg_f32_kernel` and the CPU
reference, but it is **not the selected path**: it is slower than the per-token
kernel on gfx1151 (see the measurements below). The op selects it only when the
caller supplies the scratch buffer of `yuuka_op.buf[12]`; the prefill graph
leaves that slot null.

With `p_r = gamma_1 .. gamma_r`, `g_r = log p_r`, and `g_c` the last cumulative
decay of a chunk of `n` tokens, the recurrence

```text
S_r = gamma_r (I - beta_r k_r k_r^T) S_{r-1} + beta_r k_r v_r^T
```

becomes, with `K/Q/V` the normalized chunk rows and `M = I + B` where
`B[r][s] = beta_r (k_r . k_s)` for `s < r`:

```text
S_C = exp(g_c) S_0 + K^T (M^-1 diag(beta) (exp(g_c - g) V - exp(g_c) K S_0))
o_r = exp(g_r) (q_r S_0 + Z (diag(beta exp(-g)) V - K S_0))_r
Z   = (Q K^T on the lower triangle) M^-1
```

The decay of a row is a scalar for the whole state matrix, so it factors out of
the triangular solve and only rows of the chunk interact. The chunk length is
fixed at 32 so the triangular system is one warp-wide inversion and the state
projection is a 32-row tile.

Three kernels run per prompt chunk:

- `gdn_prepare_rows_f32_kernel` normalizes the q and k segments of every row and
  replaces the raw alpha and beta projections with the log decay and the sigmoid
  gate. It consumes `conv`, `alpha`, and `beta` in place; the callers treat them
  as scratch.
- `gdn_prep_f32_kernel` runs once per (chunk, head) and writes `M^-1` and `Z`.
  One thread owns one column of the system and a quarter of the state
  dimension, so a dot product keeps 32 accumulators live and resolves with two
  shuffles. The inverse is a column-wise forward substitution on one wave.
- `gdn_scan_f32_kernel` walks the chunks of one head in order with one state
  column block per workgroup (32 of 128 columns, 8 column groups by 8 state
  parts, two columns per thread). The state slice stays in registers, the state
  projections reduce across the eight parts of a column group with shuffles,
  and the triangular products hold only the thread's own rows. Chunk-row tiles
  are staged in LDS as `tile[(i / 16) * 8 + (i % 16)][r]` so the eight parts of
  a column read windows that differ by the padded stride and land on different
  banks.

Measured with `build/yuuka_hip_prefill_sweep`-style isolation
(`/tmp/bench_gdn`, one op in a plan, 24 layers, 512 rows): the chunked path
takes 3.10 ms per layer, the per-token split-register path 2.01 ms. Per kernel,
the scan is 1.73 ms, the prep 0.85 ms, and the existing convolution, gated
RMSNorm, and preparation kernels add 0.36 ms. The scan reaches 0.4 IPC per CU:
it is LDS-latency bound, with 106-164 VGPRs and three 8-wave workgroups resident
per CU against a 47-67 cycle LDS latency, and it issues 2.7M instructions per CU
per layer. Every product is an FP32 FMA-pipe loop; the reference implementation
instead runs its dense products on the FP16 WMMA path (`mmb_dense`), which is
the step that would make this form competitive.

Correctness: `tests/hip/prefill_test.cpp` runs both paths against the same CPU
reference (chunked maximum absolute difference `1.6e-7`, per-token `2.1e-7`) and
checks that a repeated execution reproduces the outputs bitwise.

### 9.2 Per-token form in decode

Decode runs one token at a time, so the chunkwise form does not apply and the
recurrence is evaluated directly. Two operators replace the five the decode plan
used before the fusion (`CONV1D`, two `L2NORM`, `GDN`, `GATED_RMSNORM`):

- `conv_l2norm_f32_kernel` applies the short convolution to all 8192 channels,
  advancing its ring state, and then the L2 norm to the q and k groups with the q
  scale. One block owns 256 channels, which is exactly two 128-wide norm groups,
  so a norm never crosses a block and needs no cross-block reduction.
- `recurrent_token_f32_kernel` owns one head per block with 512 threads: four
  thread groups share a column of the state, each holding 32 of its 128 rows in
  registers. The block loads the column once, forms `S^T k` with the decay applied
  to the sum (because `S^T k` over the decayed state is `decay * (S^T k)` over the
  loaded state), exchanges the four partials through shared memory, forms
  `d = (v - S^T k) * beta`, and then applies the decay, the rank-1 update, and
  `S^T q` in one pass over the registers before writing the column back once. The
  gated RMSNorm of the head is folded into the same block because its reduction is
  over that head's own values: `out` receives the normalized row and the raw
  delta-net row is not written.

The state layout is unchanged, so the chunked prefill path and the decode path
consume the same buffers; nothing about the prefill operators moved.

Numerics. The five-pass kernel rounds the decayed state to memory and forms
`S^T k` from it; the fused kernel forms `decay * (S^T k)` with one rounding and
applies the update as a fused multiply-add. The column sums are also reassociated
across the four thread groups. `tests/hip/deltanet_test.cpp` bounds both device
paths against an independent host implementation of the documented order of
operations: 3e-8 on the normalized output and 7e-9 on the state between the two
device paths, over three tokens with a decay of about 0.75. The decay is chosen
far from one on purpose: with a decay of 0.997 a missing factor on `S^T k`
produces a 1.6e-5 difference, and with 0.75 it produces 2.6e-4, which the 1e-5
bound rejects.

Measured design choice ([benchmarks.md](benchmarks.md#gated-delta-net)): the
register-resident form
reaches 375 GB/s of state traffic against 125 GB/s for a shared-memory-resident
form of the same recurrence, because a 64 KB state fills the workgroup shared
budget and leaves no room for the normalization; 512 threads beat 256 by 6% and
beat a plain read-and-write copy of the same bytes, which makes memory-level
parallelism per lane the limiter rather than bandwidth.

SwiGLU remains an elementwise pass over the chunk. Fusing it into the GEMM
epilogue, together with fusing the gate/up GEMMs themselves, is a candidate;
no fused implementation is claimed. Every
fusion must be evaluated on all of:

- kernel launch count;
- VGPR usage and occupancy;
- LDS consumption;
- memory traffic;
- end-to-end latency, not just isolated kernel time.

Over-fusion can reduce occupancy or increase register pressure enough to lose
performance. A fused kernel that wins on one shape may lose on another; retain
the unfused path as a fallback when measurements justify it.

## 10. Dispatch and tuning

### 10.1 Shape signature

Dispatch decisions and plan validation use these shape and capability fields:

```text
operator, M, N, K, input dtype, weight dtype, batch, sequence class, backend
```

There is no generic kernel registry in this release. Go selectors map the
fields they need to a concrete path, and plan construction validates supported
dtypes, row layout, buffer ranges, and dimensions before submission. Lookup is
per plan, not per tensor per token.

### 10.2 Dispatch selection and validation

Dispatch is recorded in one explicit Go selector:
`backend/hip.SelectPrefillPath` chooses batched GEMV, custom rocWMMA, or the
rocBLAS FP16 baseline from dtype, rows, and the input/output widths.
The decode plan decides the fused matvec forms when it is built, not at run time:
`Model.decodeMatmulResidual` takes the residual epilogue when the streaming route
is selected, and `Model.decodeMatmulPair` takes the two-matrix form when both
shapes are on the streaming route, the two weight dtypes are equal, and — for the
SwiGLU epilogue — the widths match. Everything else keeps the separate
operations, and `Options.NoResidualFusion` / `Options.NoPairFusion` restore them
for differential measurement.

The decode plan builds a recurrent layer from `RECURRENT_TOKEN` and
`CONV_L2NORM` when the model's geometry is inside what those kernels are compiled
for (a 128-wide state and 256-aligned channel counts) and from the five separate
operators otherwise; nothing is decided at run time, and a fused operation whose
geometry is outside that set is rejected instead of routed elsewhere.
`Options.LegacyDeltaNet` forces the unfused plan for the differential test.

Single-token decode submits one matvec per matrix. `SelectDecodeGemvPath`
chooses between the per-row GEMV and the streaming kernel per operation
(section 5.3), and the output projection takes the same rule; the int8 variant is
reachable through the same selector behind a flag that defaults to off. There is
no separate runtime GEMV batch selector.

Each plan addition validates the dtype, quantization row layout, shape
constraints, and buffer ranges. Unsupported combinations fail during planning;
there is no size-keyed registry, scratch registry, or graph-compatibility
registry to consult at runtime.

### 10.3 Offline tuning

Dispatch decisions are offline. Thresholds measured on gfx1151 are encoded in
`backend/hip.SelectPrefillPath`, keyed by dtype, shape class, and row class, and
recorded with the measurement environment in [benchmarks.md](benchmarks.md).
There is no runtime autotuner. Adding a device requires re-running the sweeps
and updating the selector and tests.

### 10.4 Fallback

If a custom kernel is unavailable, unsupported for a shape, or slower than the
library baseline, the planner selects the documented fallback. Silent numerical
substitution is forbidden. A fallback must be visible in diagnostics and
benchmarks.

## 11. Numerical requirements

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

## 12. Kernel acceptance criteria

A kernel is not complete until:

- [x] It matches the CPU reference on representative shapes and buffer-boundary
      cases; randomized and non-aligned shape sweeps cover the GEMV, GEMM,
      prefill, and MoE families.
- [x] Its numerical tolerance is documented and tested.
- [x] Its relevant shape classes are covered by a benchmark or profile.
- [x] Its VGPR, LDS, scratch, and occupancy figures are recorded for the Phase
      0-3 kernel families (`scripts/kernel_resources.sh`, results in
      [benchmarks.md](benchmarks.md)).
- [x] New Phase 5 kernels have recorded VGPR, LDS, and occupancy figures for the
      GEMV, WMMA, and MoE variants (`scripts/kernel_resources.sh`, results in
      [benchmarks.md](benchmarks.md)).
- [x] The planner selects it; quantized GEMV and GEMM have measured fallbacks.
- [x] It is integrated into an end-to-end model run.
- [x] It does not regress a previously accepted shape class.
